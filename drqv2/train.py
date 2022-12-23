import os
from os.path import join as pJoin
from pathlib import Path
from warnings import filterwarnings  # noqa

import numpy as np
import torch
from params_proto.neo_proto import PrefixProto

from drqv2.replay_buffer import Replay
from . import utils
from .config import Args, Agent
from .notifier.slack_sender import slack_sender


class Progress(PrefixProto, cli=False):
    step = 0
    episode = 0
    wall_time = 0
    frame = 0


def save_checkpoint(agent, replay_cache_dir):
    from ml_logger import logger

    replay_checkpoint = os.path.join(Args.checkpoint_root, logger.prefix, 'replay.tar')

    logger.save_torch(agent, path='agent.pt')
    logger.duplicate("metrics.pkl", "metrics_latest.pkl")
    logger.upload_dir(replay_cache_dir, replay_checkpoint)
    # Save the progress.pkl last as a fail-safe. To make sure the checkpoints are saving correctly.
    logger.log_params(Progress=vars(Progress), path="progress.pkl", silent=True)


def load_checkpoint(replay_cache_dir):
    from ml_logger import logger

    replay_checkpoint = os.path.join(Args.checkpoint_root, logger.prefix, 'replay.tar')

    agent = logger.load_torch(path='agent.pt')
    logger.duplicate("metrics_latest.pkl", to="metrics.pkl")
    logger.download_dir(replay_checkpoint, replay_cache_dir)

    return agent, logger.read_params(path="progress.pkl")


def eval(env, agent, global_step, to_video=None):
    from ml_logger import logger

    step, total_reward = 0, 0
    for episode in range(Args.num_eval_episodes):
        obs = env.reset()
        frames = []
        done = False
        while not done:
            with torch.no_grad(), utils.eval_mode(agent):
                # todo: remove global_step, replace with random-on, passed-in.
                action = agent.act(obs, global_step, eval_mode=True)
            obs, reward, done, info = env.step(action)
            if episode == 0 and to_video:
                # todo: use gym.env.render('rgb_array') instead
                frames.append(env.physics.render(height=256, width=256, camera_id=0))
            total_reward += reward
            step += 1

        if episode == 0 and to_video:
            logger.save_video(frames, to_video)

    logger.log(episode_reward=total_reward / episode, episode_length=step * Args.action_repeat / episode)


def train(train_env, eval_env, agent, replay, progress: Progress, time_limit=None):
    from ml_logger import logger

    init_transition = dict(
        obs=None, reward=0.0, done=False, discount=1.0,
        action=np.full(eval_env.action_space.shape, 0.0, dtype=eval_env.action_space.dtype)
    )

    episode_step, episode_reward = 0, 0
    obs = train_env.reset()
    transition = {**init_transition, 'obs': obs}
    replay.storage.add(**transition)
    done = False
    for progress.step in range(progress.step, Args.train_frames // Args.action_repeat + 1):
        progress.wall_time = logger.since('start')
        progress.frame = progress.step * Args.action_repeat

        if done:
            progress.episode += 1

            # log stats
            episode_frame = episode_step * Args.action_repeat
            logger.log(fps=episode_frame / logger.split('episode'),
                       episode_reward=episode_reward,
                       episode_length=episode_frame,
                       buffer_size=len(replay.storage))

            # reset env
            obs = train_env.reset()
            done = False
            transition = {**init_transition, 'obs': obs}
            replay.storage.add(**transition)
            # try to save snapshot
            if time_limit and logger.since('run') > time_limit:
                logger.print(f'local time_limit: {time_limit} (sec) has reached!')
                raise TimeoutError

            episode_step, episode_reward = 0, 0

        # try to evaluate
        if logger.every(Args.eval_every_frames // Args.action_repeat, key="eval"):
            with logger.Prefix(metrics="eval"):
                path = f'videos/{progress.step * Args.action_repeat:09d}_eval.mp4'
                eval(eval_env, agent, progress.step, to_video=path if Args.save_video else None)
                logger.log(**vars(progress))
                logger.flush()

        # sample action
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, progress.step, eval_mode=False)

        # try to update the agent
        done_warming_up = progress.step * Args.action_repeat > Args.num_seed_frames
        # if logger.every(Args.update_freq, key="update") and is_warming_up:
        #     agent.update(replay.iterator, progress.step)  # checkpoint.step is used for scheduling
        if logger.every(Args.log_freq, key="log", start_on=1) and done_warming_up:
            logger.log_metrics_summary(vars(progress), default_stats='mean')
        if logger.every(100, key='checkpoint', start_on=1) and done_warming_up:
            # if logger.every(Args.checkpoint_freq, key='checkpoint', start_on=1) and done_warming_up:
            save_checkpoint(agent, replay.cache_dir)

        # take env step
        obs, reward, done, info = train_env.step(action)
        episode_reward += reward

        # TODO: is it ok to always use discount = 1.0 ?
        # we should actually take a look at time_step.discount
        transition = dict(obs=obs, reward=reward, done=done, discount=1.0, action=action)
        replay.storage.add(**transition)
        episode_step += 1


# NOTE: This wrapper will do nothing unless $SLACK_WEBHOOK_URL is set
webhook_url = os.environ.get("SLACK_WEBHOOK_URL", None)


@slack_sender(
    webhook_url=webhook_url,
    channel="rl-under-distraction-job-status",
    progress=Progress,
    ignore_exceptions=(TimeoutError,)
)
def main(**kwargs):
    from ml_logger import logger, RUN
    from drqv2.drqv2 import DrQV2Agent
    from drqv2.env_helpers import get_env
    import shutil
    # get the directory where this file is located

    from warnings import simplefilter  # noqa
    simplefilter(action='ignore', category=DeprecationWarning)

    logger.start('run')

    assert logger.prefix, "you will overwrite the experiment root"

    try:  # completion protection
        assert logger.read_params('job.completionTime')
        logger.print(f'job.completionTime is set. This job seems to have been completed already.')
        return
    except KeyError:
        pass

    replay_cache_dir = Path(Args.tmp_dir) / logger.prefix / 'replay'
    shutil.rmtree(replay_cache_dir, ignore_errors=True)
    replay_checkpoint = os.path.join(Args.checkpoint_root, logger.prefix, 'replay.tar')
    snapshot_dir = pJoin(Args.checkpoint_root, logger.prefix)

    # logger.remove('progress.pkl')
    if logger.glob('progress.pkl'):

        Args._update(**logger.read_params("Args"))
        Agent._update(**logger.read_params('Agent'))

        agent, progress_cache = load_checkpoint(replay_checkpoint)

        Progress._update(progress_cache)
        logger.start('start', 'episode')
        logger.timer_cache['start'] = logger.timer_cache['start'] - Progress.wall_time

    else:
        logger.print("Start training from scratch.")
        # load parameters
        Args._update(kwargs)
        Agent._update(kwargs)
        logger.log_params(Args=vars(Args), Agent=vars(Agent))

        # todo: this needs to be fixed.
        logger.log_text("""
            keys:
            - Args.env_name
            - Args.seed
            charts:
            - yKey: eval/episode_reward/mean
              xKey: step
            - yKey: episode_reward/mean
              xKey: step
            - yKey: fps/mean
              xKey: wall_time
            - yKeys: ["batch_reward/mean", "critic_loss/mean"]
              xKey: step
            """, ".charts.yml", overwrite=True, dedent=True)

        logger.start('start', 'episode')

    utils.set_seed_everywhere(Args.seed)
    train_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed)
    eval_env = get_env(Args.env_name, Args.frame_stack, Args.action_repeat, Args.seed)

    if 'agent' not in locals():
        agent = DrQV2Agent(obs_shape=train_env.observation_space.shape,
                           action_shape=train_env.action_space.shape,
                           device=Args.device, **vars(Agent))

    replay = Replay(cache_dir=replay_cache_dir)

    # Load from local file
    assert logger.prefix, "you will overwrite the experiment root with an empty logger.prefix"
    train(train_env, eval_env, agent, replay, progress=Progress)


if __name__ == '__main__':
    main()
