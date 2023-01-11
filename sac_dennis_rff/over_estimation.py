# from sac_rff import dmc2gym
import gym
import numpy as np
import torch
from params_proto import PrefixProto
from tqdm import tqdm

from .config import Args
from .sim_states.dmc_helper import set_sim_state
from .sac import make_env


class ValueEstimate(PrefixProto):
    checkpoint_path: str = "../"

    @classmethod
    def __init__(cls, *args, **kwargs):
        super(*args, **kwargs)


# if __name__ == '__main__':
#     value_estimate = ValueEstimate({})
#     print(*vars(value_estimate).items(), sep='\n')
#     exit()


def eval_agent_value_estimation(env, agent, sim_state, orig_state=None, gamma=0.99):
    # have to call reset first.
    env.reset()
    state = set_sim_state(env, sim_state)

    # Removing this for now
    # if orig_state is not None:
    #     assert not (orig_state - state).all(), "state mismatch"

    action = agent.act(state, sample=False)

    s_t = torch.Tensor(state)[None, :].to(Args.device)
    a_t = torch.Tensor(action)[None, :].to(Args.device)
    q1, _ = agent.critic(s_t, a_t)
    value_estimate = q1.item()

    episode_reward, done = 0, False
    t = 0
    while not done:
        state, reward, done, _ = env.step(action)
        episode_reward += reward*(gamma**t)
        t += 1
        action = agent.act(state, sample=False)

    return value_estimate, episode_reward


def main(**kwargs):
    from ml_logger import logger

    Args._update(kwargs)
    ValueEstimate._update(kwargs)

    logger.log_params(Args=vars(Args), ValueEstimate=vars(ValueEstimate))
    logger.job_started()

    logger.log_text("""
        charts:
        - yKey: [ep_return/mean, value_estimate/mean]
          xKey: epoch
        """, filename=".charts.yml", dedent=True, overwrite=True)

    # Environment
    print("making environment")
    env = make_env(Args.env_name, seed=Args.seed,
             from_pixels=Args.from_pixels,
             dmc=Args.dmc,
             image_size=Args.image_size,
             frame_stack=Args.frame_stack,
             normalize_obs=Args.normalize_obs,
             obs_bias=Args.obs_bias,
             obs_scale=Args.obs_scale)

    torch.manual_seed(Args.seed)
    np.random.seed(Args.seed)

    # print('removing existing experiment logs')
    # with logger.Sync():
    #     logger.remove('.')
    # print('done removing existing experiment logs')

    # Evaluation Loop
    for epoch in tqdm([1500, *range(100_000, 1000_001, 100_000)], leave=False):
    # for epoch in tqdm(list(range(1100_000, 2000_001, 100_000)), leave=False):

        with logger.Prefix(ValueEstimate.checkpoint_path):
            agent = logger.load_torch(Args.checkpoint_root + logger.prefix + f"/value_estimate/agent_{epoch}.pt", map_location=Args.device)
            agent.to(Args.device)

            ep_samples, = logger.load_pkl(Args.checkpoint_root + logger.prefix + f"/value_estimate/samples_{epoch}.pkl")

        for obs, sim_state in tqdm(ep_samples):
            logger.start('start', 'episode')
            value_estimate, ep_return = eval_agent_value_estimation(env, agent, sim_state=sim_state, orig_state=obs, gamma=Args.gamma)
            logger.store_metrics(value_estimate=value_estimate, ep_return=ep_return)

        logger.save_pkl(logger.summary_cache.data, f"results/estimate_{epoch:08d}.pkl")
        # this flushes the summary cache, and saves the statistics of the value estimate and the ep_return
        logger.log_metrics_summary(key_values={'epoch': epoch})
        logger.job_running()

    env.close()
