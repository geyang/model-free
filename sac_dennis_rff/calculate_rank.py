import gym
import numpy as np
import torch
from params_proto.neo_proto import PrefixProto
from tqdm import tqdm

from .config import Args
from .sac import make_env
import rff_rank


class RankCalculate(PrefixProto):
    checkpoint_path: str = "../"

    @classmethod
    def __init__(cls, *args, **kwargs):
        super(*args, **kwargs)


def calc_rank_from_rb(agent, rb, device, num_samples=10000):
    from ml_logger import logger

    rb.obses = rb.obses[~np.any(np.isnan(rb.actions), axis=1)]
    rb.actions = rb.actions[~np.any(np.isnan(rb.actions), axis=1)]
    rb.obses = rb.obses[~np.any((np.abs(rb.actions)>1), axis=1)]
    rb.actions = rb.actions[~np.any((np.abs(rb.actions)>1), axis=1)]

    inds = np.arange(rb.obses.shape[0])
    np.random.shuffle(inds)
    obses = rb.obses[inds[:num_samples]]
    acts = rb.actions[inds[:num_samples]]
    obs_tensor = torch.as_tensor(obses).float().to(device)
    action_tensor = torch.as_tensor(acts).float().to(device)

    obs_action_tensor = torch.cat([obs_tensor, action_tensor], dim=-1)

    if agent.critic.use_rff:
        obs_action_tensor = agent.critic.critic_1_rff(obs_action_tensor)

    with logger.time("effective rank"):
        rank = rff_rank.H_srank(xs=obs_action_tensor, net=agent.critic.Q1)

    logger.print(f'eval/q_rank: {rank}')
    return rank

def get_bias(module):
    params = []
    for k,v in module.state_dict().items():
        if 'bias' in k:
            params.append(v)
    return params

def get_weight(module):
    params = []
    for k,v in module.state_dict().items():
        if 'bias' not in k:
            params.append(v)
    return params

def main(**kwargs):
    from ml_logger import logger

    Args._update(kwargs)
    RankCalculate._update(kwargs)

    logger.log_params(Args=vars(Args), RankCalculate=vars(RankCalculate))
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

    print('removing existing experiment logs')
    with logger.Sync():
        logger.remove('W_diff.pkl')
    print('done removing existing experiment logs')

    # Evaluation Loop
    with logger.Prefix(RankCalculate.checkpoint_path):
        # replay_buffer = logger.load_torch(Args.checkpoint_root + logger.prefix + "/checkpoint/replay_buffer.pkl", map_location='cpu')
        agent_init = logger.load_torch(Args.checkpoint_root + logger.prefix + f"/value_estimate/agent_0.pt", map_location='cpu')
        if agent_init.critic.use_rff:
            # agent_init_param = torch.cat([param.view(-1) for param in agent_init.critic.critic_1_rff.parameters()])
            # agent_init_param = torch.cat([param.view(-1) for param in agent_init.critic.Q1.parameters()])
            # agent_init_param = torch.cat([param.view(-1) for param in list(agent_init.critic.critic_1_rff.parameters()) + list(agent_init.critic.Q1.parameters())])
            # agent_init_param = torch.cat([param.view(-1) for param in get_bias(agent_init.critic.critic_1_rff)])
            # agent_init_param = torch.cat([param.view(-1) for param in get_bias(agent_init.critic.Q1)])
            # agent_init_param = torch.cat([param.view(-1) for param in get_weight(agent_init.critic.critic_1_rff)])
            agent_init_param = torch.cat([param.view(-1) for param in get_weight(agent_init.critic.Q1)])
        else:
            # agent_init_param = torch.cat([param.view(-1) for param in list(agent_init.critic.Q1.parameters())[:2]])
            # agent_init_param = torch.cat([param.view(-1) for param in list(agent_init.critic.Q1.parameters())[2:]])
            # agent_init_param = torch.cat([param.view(-1) for param in agent_init.critic.Q1.parameters()])
            # agent_init_param = torch.cat([param.view(-1) for param in get_bias(agent_init.critic.Q1)[:1]])
            # agent_init_param = torch.cat([param.view(-1) for param in get_bias(agent_init.critic.Q1)[1:]])
            # agent_init_param = torch.cat([param.view(-1) for param in get_weight(agent_init.critic.Q1)[:1]])
            agent_init_param = torch.cat([param.view(-1) for param in get_weight(agent_init.critic.Q1)[1:]])

    if Args.env_name == 'dmc:Humanoid-run-v1':
        end_epoch = 2000_001
    else:
        end_epoch = 1000_001

    for epoch in tqdm([1500, *range(100_000, end_epoch, 100_000)], leave=False):

        with logger.Prefix(RankCalculate.checkpoint_path):
            agent = logger.load_torch(Args.checkpoint_root + logger.prefix + f"/value_estimate/agent_{epoch}.pt", map_location='cpu')

            if agent.critic.use_rff:
                # agent_param = torch.cat([param.view(-1) for param in agent.critic.critic_1_rff.parameters()])
                # agent_param = torch.cat([param.view(-1) for param in agent.critic.Q1.parameters()])
                # agent_param = torch.cat([param.view(-1) for param in list(agent.critic.critic_1_rff.parameters()) + list(agent.critic.Q1.parameters())])
                # agent_param = torch.cat([param.view(-1) for param in get_bias(agent.critic.critic_1_rff)])
                # agent_param = torch.cat([param.view(-1) for param in get_bias(agent.critic.Q1)])
                # agent_param = torch.cat([param.view(-1) for param in get_weight(agent.critic.critic_1_rff)])
                agent_param = torch.cat([param.view(-1) for param in get_weight(agent.critic.Q1)])
            else:
                # agent_param = torch.cat([param.view(-1) for param in list(agent.critic.Q1.parameters())[:2]])
                # agent_param = torch.cat([param.view(-1) for param in list(agent.critic.Q1.parameters())[2:]])
                # agent_param = torch.cat([param.view(-1) for param in agent.critic.Q1.parameters()])
                # agent_param = torch.cat([param.view(-1) for param in get_bias(agent.critic.Q1)[:1]])
                # agent_param = torch.cat([param.view(-1) for param in get_bias(agent.critic.Q1)[1:]])
                # agent_param = torch.cat([param.view(-1) for param in get_weight(agent.critic.Q1)[:1]])
                agent_param = torch.cat([param.view(-1) for param in get_weight(agent.critic.Q1)[1:]])

            # agent_param = torch.cat([param.view(-1) for param in agent.parameters()])
            # agent.to(Args.device)

        weight_diff = torch.norm(agent_param - agent_init_param, p=2) #/(torch.norm(agent_init_param, p=2) + 1e-5)
        weight_diff = weight_diff.cpu().item()
        # rank = calc_rank_from_rb(agent=agent, rb=replay_buffer, device=Args.device)
        logger.log_metrics(epoch=epoch, weight_diff=weight_diff, file='W_diff.pkl', flush=True)

        logger.job_running()

    env.close()
