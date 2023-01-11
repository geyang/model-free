import torch
from params_proto import ParamsProto, Proto, Flag


class Args(ParamsProto):
    """
    PPO arguments
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    algo = Proto('a2c', help='algorithm to use: a2c | ppo | acktr')
    gail = Proto(False, help='do imitation learning with gail')
    gail_experts_dir = Proto('./gail_experts', help='directory that contains expert demonstrations for gail')
    gail_batch_size = Proto(128, help='gail batch size')
    gail_epoch = Proto(5, help='gail epochs')
    lr = Proto(7e-4, help='learning rate')
    eps = Proto(1e-5, help='RMSprop optimizer epsilon')
    alpha = Proto(0.99, help='RMSprop optimizer apha')
    gamma = Proto(0.99, help='discount factor for rewards')
    use_gae = Proto(False, help='use generalized advantage estimation')
    gae_lambda = Proto(0.95, help='gae lambda parameter')
    entropy_coef = Proto(0.01, help='entropy term coefficient')
    value_loss_coef = Proto(0.5, help='value loss coefficient')
    max_grad_norm = Proto(0.5, help='max norm of gradients')
    seed = Proto(1, help='random seed')
    num_processes = Proto(16, help='how many training CPU processes to use')
    num_steps = Proto(5, help='number of forward steps in A2C')
    ppo_epoch = Proto(4, help='number of ppo epochs')
    num_mini_batch = Proto(32, help='number of batches for ppo')
    clip_param = Proto(0.2, help='ppo clip parameter')
    log_interval = Proto(10, help='log interval, one log per n updates')
    save_interval = Proto(100, help='save interval, one save per n updates')
    eval_interval = Proto(None, help='eval interval, one eval per n updates')
    num_env_steps = Proto(10e6, help='number of environment steps to train')
    env_name = Proto('PongNoFrameskip-v4', help='environment to train on')
    log_dir = Proto('/tmp/gym/', help='directory to save agent logs')
    save_dir = Proto('./trained_models/', help='directory to save agent logs')
    no_cuda = Proto(False, help='disables CUDA training')
    use_proper_time_limits = Proto(False, help='compute returns taking into account time limits')
    recurrent_policy = Proto(False, help='use a recurrent policy')
    use_linear_lr_decay = Proto(False, help='use a linear schedule on the learning rate')


    # assert args.algo in ['a2c', 'ppo', 'acktr']
    # if args.recurrent_policy:
    #     assert args.algo in ['a2c', 'ppo'], \
    #         'Recurrent policy is not implemented for ACKTR'
    #
    # return args
