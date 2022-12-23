import os

import torch

from ppo.a2c_ppo_acktr import algo, utils
from ppo.a2c_ppo_acktr.algo import gail
from ppo.a2c_ppo_acktr.config import Args
from ppo.a2c_ppo_acktr.envs import make_vec_envs
from ppo.a2c_ppo_acktr.model import Policy
from ppo.a2c_ppo_acktr.storage import RolloutStorage
from ppo.evaluation import evaluate


def main(deps=None):
    """
    PPO Main Training Script
    :param deps: params-proto config dictionary
    """
    from ml_logger import logger

    Args._update(deps)
    logger.log_params(Args=vars(Args))

    torch.manual_seed(Args.seed)

    if Args.device == 'cuda':
        torch.cuda.manual_seed_all(Args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # is slower

    log_dir = os.path.expanduser(Args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = Args.device

    envs = make_vec_envs(Args.env_name, Args.seed, Args.num_processes,
                         Args.gamma, Args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': Args.recurrent_policy})
    actor_critic.to(device)

    if Args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            Args.value_loss_coef,
            Args.entropy_coef,
            lr=Args.lr,
            eps=Args.eps,
            alpha=Args.alpha,
            max_grad_norm=Args.max_grad_norm)
    elif Args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            Args.clip_param,
            Args.ppo_epoch,
            Args.num_mini_batch,
            Args.value_loss_coef,
            Args.entropy_coef,
            lr=Args.lr,
            eps=Args.eps,
            max_grad_norm=Args.max_grad_norm)
    elif Args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, Args.value_loss_coef, Args.entropy_coef, acktr=True)

    if Args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(Args.gail_experts_dir, f"trajs_{Args.env_name.split('-')[0].lower()}.pt")

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > Args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=Args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(Args.num_steps, Args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    num_updates = int(Args.num_env_steps) // Args.num_steps // Args.num_processes
    for epoch in range(num_updates + 1):

        if Args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, epoch, num_updates,
                agent.optimizer.lr if Args.algo == "acktr" else Args.lr)

        for step in range(Args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    logger.store_metrics({"train/reward": info['episode']['r']})

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0]
                                           for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if Args.gail:
            if epoch >= 10:
                envs.venv.eval()

            gail_epoch = Args.gail_epoch
            if epoch < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts, utils.get_vec_normalize(envs)._obfilt)

            for step in range(Args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], Args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, Args.use_gae, Args.gamma,
                                 Args.gae_lambda, Args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        total_num_steps = (epoch + 1) * Args.num_processes * Args.num_steps
        logger.store_metrics({"losses/dist_entropy": dist_entropy,
                              "losses/value_loss": value_loss,
                              "losses/action_loss": action_loss,
                              })

        # save for every interval-th episode or for the last epoch
        if epoch % Args.save_interval == 0 or epoch == num_updates - 1 and Args.save_dir != "":
            save_path = os.path.join(Args.save_dir, Args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, Args.env_name + ".pt"))

        if epoch % Args.log_interval == 0:
            logger.log_metrics_summary(key_values=dict(steps=total_num_steps, epoch=epoch))
            # end = time.time()
            # print(f"Updates {epoch}, num timesteps {total_num_steps}, "
            #       f"FPS {int(total_num_steps / (end - start))} \n "
            #       f"Last {len(episode_rewards)} training episodes: "
            #       f"mean/median reward {np.mean(episode_rewards):.1f}/{np.median(episode_rewards):.1f}, "
            #       f"min/max reward {np.min(episode_rewards):.1f}/{np.max(episode_rewards):.1f}\n")

        if Args.eval_interval and epoch % Args.eval_interval == 0:
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, Args.env_name, Args.seed, Args.num_processes, eval_log_dir, device)
            logger.log_metrics_summary(key_values=dict(steps=total_num_steps, epoch=epoch))


if __name__ == "__main__":
    main()
