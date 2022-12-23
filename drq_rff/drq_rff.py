from contextlib import ExitStack
from copy import deepcopy

import gym
# import dmc2gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import pickle

from . import utils
from rff_kernels import models
from .replay_buffer import ReplayBuffer
from .config import Args, Encoder, Actor, Critic, Agent
from params_proto.neo_proto import PrefixProto


class Progress(PrefixProto, cli=False):
    step = 0
    episode = 0


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class View(nn.Module):
    def __init__(self, *dims, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dims = dims

    def forward(self, x):
        if self.batch_first:
            return x.view(-1, *self.dims)
        else:
            return x.view(*self.dims)


class Join(nn.Module):
    def __init__(self, *modules):
        """Join Module

        :param modules: assumes that each module takes only 1 input.
        """
        super().__init__()
        self.modules = modules

    def forward(self, *inputs):
        return torch.cat([net(x) for x, net in zip(inputs, self.modules)], dim=1)


class YComb(nn.Module):
    def __init__(self, left, right, split):
        """Join Module

        :param modules: assumes that each module takes only 1 input.
        """
        super().__init__()
        self.left = left
        self.right = right
        self.split = split

    def forward(self, inputs):
        left_input, right_input = inputs.split(self.split, dim=1)
        return torch.cat([self.left(left_input), self.right(right_input)], dim=1)


class Identity(nn.Module):
    out_features: int

    def __init__(self, in_features: int):
        super().__init__()
        self.out_features = in_features

    def forward(self, obs, **_):
        return obs


class MLP(nn.Sequential):
    def __init__(self, in_features, out_features, hidden_features=None, hidden_layers=0, use_rff=False, input_rff=None, use_dense=False,
                 out_linear=False):
        self.input_rff = input_rff
        self.use_rff = use_rff

        if self.use_rff:
            assert in_features == self.input_rff.in_features
            in_features = self.input_rff.out_features

        if not hidden_layers:
            layers = [
                nn.Linear(in_features, out_features),
                nn.ReLU()
            ]
        else:
            if isinstance(hidden_features, int):
                layers = [
                    nn.Linear(in_features, hidden_features),
                    nn.ReLU()
                ]
                dim_units = [hidden_features] * (hidden_layers - 1) + [out_features]
            else:
                assert isinstance(hidden_features, list) or isinstance(hidden_features, tuple)
                assert len(hidden_features) == hidden_layers
                layers = [
                    nn.Linear(in_features, hidden_features[0]),
                    nn.ReLU(),
                ]
                dim_units = hidden_features[1:] + [out_features]

            for dim in dim_units:
                if use_dense:
                    layers += [
                        nn.Linear(layers[-2].out_features + in_features, dim),
                        nn.ReLU(),
                    ]
                else:
                    layers += [
                        nn.Linear(layers[-2].out_features, dim),
                        nn.ReLU(),
                    ]
        super().__init__(*layers[:-1] if out_linear else layers)
        self.in_features = in_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.use_dense = use_dense

    def forward(self, obs, detach=False):
        with torch.no_grad() if detach else ExitStack():
            num_layers = len(self)

            if self.use_rff:
                # project obs into random fourier basis
                obs = self.input_rff(obs)

            h = obs
            for (i, module) in enumerate(self):
                h = module(h)
                if self.use_dense and isinstance(module, nn.ReLU) and i < num_layers - 1:
                    # Ensures not to concatenate input to output layer
                    h = torch.cat([h, obs], dim=1)
            return h

    def copy_weights_from(self, source):
        """Tie layers"""
        from ml_logger import logger
        for i, (src, trg) in enumerate(zip(source, self)):
            try:
                utils.tie_weights(src, trg)
            except:
                logger.print(f"layer{i}: {source[i]} does not contain weight")


class DummyEncoder(nn.Module):
    out_features: int

    def __init__(self, out_feat, use_rff=False, input_rff=None):
        super().__init__()
        self.use_rff = use_rff
        self.input_rff = input_rff

        if self.use_rff:
            assert out_feat == self.input_rff.in_features
            out_feat = self.input_rff.out_features

        self.out_features = out_feat

    def forward(self, obs, **_):
        if self.use_rff:
            return self.input_rff(obs)
        return obs

    def copy_weights_from(self, source):
        pass


class DrQActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, encoder, action_dim, hidden_features, hidden_layers, log_std_bounds,
                 use_dense=False):
        super().__init__()
        self.encoder = encoder
        self.log_std_bounds = log_std_bounds
        self.trunk = MLP(self.encoder.out_features, out_features=2*action_dim,
                         hidden_layers=hidden_layers, hidden_features=hidden_features, use_dense=use_dense,
                         out_linear=True)
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        return utils.SquashedNormal(mu, std)


class DrQCritic(nn.Module):
    """Critic network, employs double Q-learning."""

    def __init__(self, encoder, action_dim, hidden_features, hidden_layers,
                 use_rff=False, action_rff=None, use_dense=False):
        super().__init__()

        self.encoder = encoder
        assert hidden_layers, "can not use zero hidden_layers."

        if use_rff:
            self.Q1 = nn.Sequential(
                YComb(Identity(self.encoder.out_features), action_rff, self.encoder.out_features),
                MLP(self.encoder.out_features + action_rff.out_features, out_features=1,
                    hidden_layers=hidden_layers, hidden_features=hidden_features, out_linear=True)
            )
            self.Q2 = nn.Sequential(
                YComb(Identity(self.encoder.out_features), action_rff, self.encoder.out_features),
                MLP(self.encoder.out_features + action_rff.out_features, out_features=1,
                    hidden_layers=hidden_layers, hidden_features=hidden_features, out_linear=True)
            )
        else:
            self.Q1 = MLP(self.encoder.out_features + action_dim, out_features=1,
                          hidden_layers=hidden_layers, hidden_features=hidden_features, use_dense=use_dense,
                          out_linear=True)
            self.Q2 = MLP(self.encoder.out_features + action_dim, out_features=1,
                          hidden_layers=hidden_layers, hidden_features=hidden_features, use_dense=use_dense,
                          out_linear=True)

        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2


class DrQAgent(nn.Module):
    """Data regularized Q: actor-critic method for learning from pixels."""

    def __init__(self, obs_shape, action_shape, action_range,
                 actor, critic, discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, share_encoder, **_):
        super().__init__()
        self.action_range = action_range
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.actor = actor
        self.critic = critic

        # tie conv layers between actor and critic
        if share_encoder:
            self.actor.encoder.copy_weights_from(self.critic.encoder)

        if self.critic_tau is None:
            self.critic_target = self.critic
        else:
            self.critic_target = deepcopy(critic)
            self.critic_target.load_state_dict(self.critic.state_dict())

        # this way this gets sent to cuda automatically
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_temperature)), requires_grad=True)  # .to(device)
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def device(self):
        return next(self.parameters()).device

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        from ml_logger import logger
        logger.store_metrics({'train/critic_loss': critic_loss})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, step):
        from ml_logger import logger

        # detach conv filters, so we don't update them with the actor loss
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # detach conv filters, so we don't update them with the actor loss
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.store_metrics({'train/actor_loss': actor_loss,
                              'train/actor_target_entropy': self.target_entropy,
                              'train/actor_entropy': -log_prob.mean()})

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        logger.store_metrics({'train/alpha_loss': alpha_loss, 'train/alpha_value': self.alpha})
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update(self, replay_buffer, step):
        from ml_logger import logger
        obs, action, reward, next_obs, not_done, _, _ = replay_buffer.sample(
            self.batch_size)

        logger.store_metrics({'train/batch_reward': reward.mean()})

        self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if self.critic_tau and step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


def make_env(env_name, seed, action_repeat, from_pixels=True, dmc=True, image_size=None, frame_stack=None,
             normalize_obs=False, obs_bias=None, obs_scale=None):
    """Helper function to create dm_control environment"""
    from ml_logger import logger

    domain_name, task_name, *_ = env_name.split(":")[-1].split('-')
    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'Quadruped' else 0

    if dmc:
        env = gym.make(env_name,
                       visualize_reward=False,
                       from_pixels=from_pixels,
                       height=image_size,
                       width=image_size,
                       frame_skip=action_repeat,
                       camera_id=camera_id)
    else:
        env = gym.make(env_name)

    if normalize_obs:
        logger.print(f'obs bias is {obs_bias}', color="green")
        logger.print(f'obs scale is {obs_scale}', color="green")

    env = utils.NormalizedBoxEnv(env, obs_mean=obs_bias, obs_std=obs_scale)

    if from_pixels and frame_stack:
        env = utils.FrameStack(env, k=frame_stack)

    env.seed(seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def train(**deps):
    from ml_logger import logger, RUN

    RUN._update(deps)
    if RUN.resume and logger.glob("checkpoint.pkl"):
        deps = logger.read_params()
    else:
        RUN.resume = False

    Args._update(deps)
    Encoder._update(deps)
    Actor._update(deps)
    Critic._update(deps)
    Agent._update(deps)

    if RUN.resume:
        logger.print("Loading from checkpoint...", color="yellow")
        logger.duplicate("metrics_latest.pkl", to="metrics.pkl")
        Progress._update(logger.read_params(path="checkpoint.pkl"))
        # note: maybe remove the error later after the run stablizes
        logger.remove("traceback.err")
        if Progress.episode > 0:  # the episode never got completed
            Progress.episode -= 1
    else:
        logger.remove('metrics.pkl', 'checkpoint.pkl', 'metrics_latest.pkl', "traceback.err")
        logger.log_params(RUN=vars(RUN), Args=vars(Args), Actor=vars(Actor), Critic=vars(Critic), Encoder=vars(Encoder))
        logger.log_text("""
            charts:
            - yKey: train/episode_reward/mean
              xKey: step
            - yKey: eval/episode_reward/mean
              xKey: step
            """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = True
    utils.set_seed_everywhere(Args.seed)

    env = make_env(Args.env_name, seed=Args.seed,
                   from_pixels=Args.from_pixels,
                   dmc=Args.dmc,
                   action_repeat=Args.action_repeat,
                   image_size=Args.image_size,
                   frame_stack=Args.frame_stack,
                   normalize_obs=Args.normalize_obs,
                   obs_bias=Args.obs_bias,
                   obs_scale=Args.obs_scale)

    eval_env = make_env(Args.env_name, seed=Args.seed,
                        from_pixels=Args.from_pixels,
                        dmc=Args.dmc,
                        action_repeat=Args.action_repeat,
                        image_size=Args.image_size,
                        frame_stack=Args.frame_stack,
                        normalize_obs=Args.normalize_obs,
                        obs_bias=Args.obs_bias,
                        obs_scale=Args.obs_scale)

    if RUN.resume:
        agent = logger.load_torch(Args.checkpoint_root, logger.prefix, 'checkpoint/agent.pkl', map_location=Args.device)
        replay_buffer = logger.load_torch(Args.checkpoint_root, logger.prefix, 'checkpoint/replay_buffer.pkl')
    else:
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]

        if Agent.use_rff:
            logger.print(f"Using RFF (type={Agent.rff_type}) with scale-{Agent.in_scale}", color="green")
            rff_class = models.RFF_dict[Agent.rff_type]
            state_rff = rff_class(obs_shape[0], Agent.state_fourier_features, scale=Agent.in_scale, init=Agent.rff_init)
            action_rff = rff_class(action_shape[0], Agent.action_fourier_features, scale=Agent.in_scale, init=Agent.rff_init)
        else:
            state_rff = None
            action_rff = None

        def get_encoder():
            if Args.from_pixels:
                """Note: conv uses hard-coded num_layers and num_filters"""
                raise NotImplementedError
            elif Encoder.dummy:
                logger.print("Using dummy encoder", color="green")
                return DummyEncoder(out_feat=obs_shape[0], use_rff=Agent.use_rff, input_rff=state_rff)
            logger.print("Encoder using mlp", color="green")
            return MLP(in_features=obs_shape[0],
                       hidden_layers=Encoder.hidden_layers,
                       hidden_features=Encoder.hidden_features,
                       out_features=Encoder.out_features,
                       use_rff=Agent.use_rff,
                       input_rff=state_rff,
                       use_dense=Encoder.use_dense,
                       )

        actor = DrQActor(
            encoder=get_encoder(),
            action_dim=action_shape[0],
            hidden_layers=Actor.hidden_layers,
            hidden_features=Actor.hidden_features,
            log_std_bounds=[-10, 2],
            use_dense=Actor.use_dense,
        )

        critic = DrQCritic(
            encoder=get_encoder(),
            action_dim=action_shape[0],
            hidden_layers=Critic.hidden_layers,
            hidden_features=Critic.hidden_features,
            use_rff=Agent.use_rff,
            action_rff=action_rff,
            use_dense=Critic.use_dense,
        )

        agent = DrQAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            critic=critic,
            actor=actor,
            **vars(Agent)
        )
        agent.to(Args.device)

        replay_buffer = ReplayBuffer(obs_shape, action_shape, Args.replay_buffer_size, Args.image_pad,
                                     pixel=Args.from_pixels, aug=Args.aug, device=Args.device)

    logger.print('now start running', color="green")

    run(env, eval_env, agent, replay_buffer,
        progress=Progress,
        train_steps=Args.train_frames // Args.action_repeat,
        seed_steps=Args.seed_frames // Args.action_repeat, **vars(Args))


def evaluate(env, agent, step, n_episode, save_video=None, compute_rank=False):
    from ml_logger import logger

    average_episode_reward = 0
    pred_q_lst = []
    true_q_lst = []
    frames = []
    if compute_rank:
        feats_1 = []
        feats_2 = []
    for episode in trange(n_episode):
        obs = env.reset()
        done = False
        episode_reward = 0
        averaged_true_q = 0
        averaged_pred_q = 0
        episode_step = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            obs_tensor = torch.as_tensor(obs[None], device=agent.device).float()
            action_tensor = torch.as_tensor(action[None], device=agent.device).float()
            # Calculating sum of predicted Q values along the trajectory
            if compute_rank:
                (q1, feat_q1), (q2, feat_q2) = agent.critic(obs_tensor, action_tensor, get_feat=True)
                feats_1.append(feat_q1.detach())
                feats_2.append(feat_q2.detach())
            else:
                q1, q2 = agent.critic(obs_tensor, action_tensor)
            averaged_pred_q += torch.min(q1, q2).item()
            obs, reward, done, info = env.step(action)
            if Args.from_pixels:
                img = obs.transpose([1, 2, 0])[:, :, :3]
            else:
                img = env.render("rgb_array", width=Args.image_size, height=Args.image_size)
            if save_video:
                frames.append(img)
            episode_reward += reward
            episode_step += 1
            # Calculating sum of Q values along the trajectory
            averaged_true_q += reward * (1 - (agent.discount ** episode_step)) / (1 - agent.discount)
        average_episode_reward += episode_reward
        # Dividing by episode step to calculate average of Q values along trajectory
        averaged_true_q = averaged_true_q / episode_step
        # Dividing by episode step to calculate average of predicted Q values along trajectory
        averaged_pred_q = averaged_pred_q / episode_step
        true_q_lst.append(averaged_true_q)
        pred_q_lst.append(averaged_pred_q)
    if save_video:
        logger.save_video(frames, save_video)
    average_episode_reward /= n_episode

    logger.store_metrics(metrics={'eval/episode_reward': average_episode_reward,
                                  'eval/avg_pred_q': np.mean(pred_q_lst),
                                  'eval/avg_true_q': np.mean(true_q_lst)})

    if compute_rank:
        feats_1 = torch.cat(feats_1, dim=0)
        feats_2 = torch.cat(feats_2, dim=0)

        with logger.time("effective rank"):
            rank_1 = get_effective_rank(feats_1, 1000)
            rank_2 = get_effective_rank(feats_2, 1000)

        logger.store_metrics(metrics={'eval/q1_rank': rank_1, 'eval/q2_rank': rank_2})


def run(env, eval_env, agent, replay_buffer, progress, seed_steps, train_steps, optim_iters,
        eval_frequency, eval_episodes, checkpoint_freq, checkpoint_root, save_video, action_repeat,
        save_final_replay_buffer, seed, report_rank, **_):
    from ml_logger import logger

    episode_reward, episode_step, done = 0, 1, True
    logger.start('episode')
    start_step = progress.step
    for progress.step in range(start_step, train_steps + 1):

        # evaluate agent periodically
        if progress.step % eval_frequency == 0:
            evaluate(eval_env, agent, progress.step, n_episode=eval_episodes,
                     save_video=f'videos/{progress.step:07d}.mp4' if save_video else None,
                     compute_rank=report_rank)

        if progress.step % checkpoint_freq == 0:
            logger.job_running()  # mark the job to be running.
            logger.print(f"saving checkpoint: {checkpoint_root}/{logger.prefix}", color="green")
            with logger.time('checkpoint.agent'):
                logger.save_torch(agent, checkpoint_root, logger.prefix, 'checkpoint/agent.pkl')
            with logger.time('checkpoint.buffer'):
                logger.save_torch(replay_buffer, checkpoint_root, logger.prefix, 'checkpoint/replay_buffer.pkl')
            logger.duplicate("metrics.pkl", "metrics_latest.pkl")
            logger.log_params(Progress=vars(progress), path="checkpoint.pkl", silent=True)

        if done:
            logger.store_metrics({'train/episode_reward': episode_reward})
            dt_episode = logger.split('episode')
            logger.log_metrics_summary(
                key_values={"episode": progress.episode,
                            "step": progress.step,
                            "frames": progress.step * action_repeat,
                            "fps": episode_step * action_repeat / dt_episode,
                            "dt_episode": dt_episode})

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            progress.episode += 1

        # sample action for data collection
        if progress.step < seed_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)

        # run training update
        if progress.step >= seed_steps:
            for _ in range(optim_iters):
                agent.update(replay_buffer, progress.step)

        next_obs, reward, done, info = env.step(action)

        # allow infinite bootstrap
        done = float(done)
        done_no_max = 0 if episode_step + 1 == env._max_episode_steps else done
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

        obs = next_obs
        episode_step += 1

    if save_final_replay_buffer:
        logger.print("saving replay buffer", color="green")
        logger.save_torch(replay_buffer, checkpoint_root, logger.prefix, 'checkpoint/replay_buffer.pkl')