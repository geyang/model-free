from contextlib import ExitStack
from copy import deepcopy

import gym
# import dmc2gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from . import utils
from .config import Args, Agent


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


class DummyEncoder(nn.Module):
    feature_dim: int

    def __init__(self, obs_shape: tuple):
        super().__init__()
        self.feature_dim = obs_shape[-1]

    def forward(self, obs, **_):
        return obs

    def copy_conv_weights_from(self, source):
        pass


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""

    def __init__(self, obs_shape: tuple, feature_dim: int):
        super().__init__()

        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.output_dim = 35
        self.output_logits = False
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            Lambda(lambda x: x / 255),
            nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            View(self.num_filters * 35 * 35),
            nn.Linear(self.num_filters * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        if self.output_logits:
            self.head.append(nn.Tanh())

    def forward(self, obs, detach=False):
        with torch.no_grad() if detach else ExitStack():
            h = self.conv(obs)
        return self.head(h)

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        from ml_logger import logger
        for i, (src, trg) in enumerate(zip(source.conv, self.conv)):
            try:
                utils.tie_weights(src, trg)
            except:
                logger.print(f"layer{i}: {source.conv[i]} does not contain weight")


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, encoder, action_shape, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()

        self.encoder = encoder

        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim, 2 * action_shape[0], hidden_depth)

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


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, encoder, action_shape, hidden_dim, hidden_depth):
        super().__init__()

        self.encoder = encoder

        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)

        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2


class DRQAgent(nn.Module):
    """Data regularized Q: actor-critic method for learning from pixels."""

    def __init__(self, obs_shape, action_shape, action_range,
                 actor, critic,
                 discount,
                 init_temperature, lr, actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size):
        super().__init__()
        self.action_range = action_range
        # self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.actor = actor
        self.critic = critic
        self.critic_target = deepcopy(critic)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie conv layers between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

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

    def update_critic(self, obs, obs_aug, action, reward, next_obs, next_obs_aug, not_done):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug, next_action_aug)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        from ml_logger import logger
        logger.store_metrics({'train/critic_loss': critic_loss})

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(logger, step)

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
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample(
            self.batch_size)

        logger.store_metrics({'train/batch_reward': reward.mean()})

        self.update_critic(obs, obs_aug, action, reward, next_obs, next_obs_aug, not_done)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)


def make_env(env_name, seed, action_repeat, from_pixels=True, image_size=None, frame_stack=None):
    """Helper function to create dm_control environment"""
    from drq import utils

    domain_name, task_name, *_ = env_name.split(":")[-1].split('-')
    # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
    camera_id = 2 if domain_name == 'Quadruped' else 0

    env = gym.make(env_name,
                   visualize_reward=False,
                   from_pixels=from_pixels,
                   height=image_size,
                   width=image_size,
                   frame_skip=action_repeat,
                   camera_id=camera_id)
    if from_pixels and frame_stack:
        env = utils.FrameStack(env, k=frame_stack)
    env.seed(seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def train(**deps):
    from . import utils
    from .drq import DRQAgent, Critic, Actor, Encoder
    from .replay_buffer import ReplayBuffer
    from ml_logger import logger

    Args._update(**deps)

    logger.log_params(Args=vars(Args))
    logger.remove('metrics.pkl')
    logger.log_text("""
        charts:
        - yKey: train/episode_reward/mean
          xKey: step
        - yKey: eval/episode_reward
          xKey: step
        """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = True
    utils.set_seed_everywhere(Args.seed)

    env = make_env(Args.env_name, seed=Args.seed,
                   from_pixels=Args.from_pixels,
                   action_repeat=Args.action_repeat,
                   image_size=Args.image_size,
                   frame_stack=Args.frame_stack)

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]

    critic = Critic(
        encoder=Encoder(obs_shape=obs_shape,
                        feature_dim=Args.feature_dim) if Args.from_pixels else DummyEncoder(obs_shape),
        action_shape=action_shape,
        hidden_dim=1024,
        hidden_depth=2)

    actor = Actor(
        encoder=Encoder(obs_shape=obs_shape,
                        feature_dim=Args.feature_dim) if Args.from_pixels else DummyEncoder(obs_shape),
        action_shape=action_shape,
        hidden_depth=2,
        hidden_dim=1024,
        log_std_bounds=[-10, 2])

    agent = DRQAgent(
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
    run(env, agent, replay_buffer,
        start_step=0,
        train_steps=Args.train_frames // Args.action_repeat,
        seed_steps=Args.seed_frames // Args.action_repeat, **vars(Args))


def evaluate(env, agent, n_episode, save_video=None):
    from drq import utils
    from ml_logger import logger

    average_episode_reward = 0
    frames = []
    for episode in trange(n_episode):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            obs, reward, done, info = env.step(action)
            if Args.from_pixels:
                img = obs.transpose([1, 2, 0])[:, :, :3]
            else:
                img = env.render("rgb_array", width=Args.image_size, height=Args.image_size)
            frames.append(img)
            episode_reward += reward
            episode_step += 1

        average_episode_reward += episode_reward
    if save_video:
        logger.save_video(frames, save_video)
    average_episode_reward /= n_episode
    logger.log(metrics={'eval/episode_reward': average_episode_reward})


def run(env, agent, replay_buffer, start_step, seed_steps, train_steps, optim_iters,
        eval_frequency, eval_episodes, save_video, action_repeat, **_):
    from drq import utils
    from ml_logger import logger

    episode, episode_reward, episode_step, done = 0, 0, 1, True

    logger.start('episode')
    for step in range(start_step, train_steps + 1):
        if done:
            # evaluate agent periodically
            if step % eval_frequency == 0:
                evaluate(env, agent, n_episode=eval_episodes,
                         save_video=f'videos/{step:07d}.mp4' if save_video else None)

            logger.store_metrics({'train/episode_reward': episode_reward})
            dt_episode = logger.split('episode')
            logger.log_metrics_summary(
                key_values={"episode": episode,
                            "step": step,
                            "frames": step * action_repeat,
                            "fps": episode_step * action_repeat / dt_episode,
                            "dt_episode": dt_episode})

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        # sample action for data collection
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)

        # run training update
        if step >= seed_steps:
            for _ in range(optim_iters):
                agent.update(replay_buffer, step)

        next_obs, reward, done, info = env.step(action)

        # allow infinite bootstrap
        done = float(done)
        done_no_max = 0 if episode_step + 1 == env._max_episode_steps else done
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

        obs = next_obs
        episode_step += 1
