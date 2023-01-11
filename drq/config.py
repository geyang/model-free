import torch

from params_proto import ParamsProto, PrefixProto


class Args(ParamsProto):
    env_name = "dmc:Cartpole-swingup-v1"
    # IMPORTANT= if action_repeat is used the effective number of env steps needs to be
    # multiplied by action_repeat in the result graphs.
    # This is a common practice for a fair comparison.
    # See the 2nd paragraph in Appendix C of SLAC= https=//arxiv.org/pdf/1907.00953.pdf
    # See Dreamer TF2's implementation= https=//github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/dreamer.py#L340
    action_repeat = 4
    # train
    train_frames = 4_000_000
    seed_frames = 4_000
    optim_iters = 1
    replay_buffer_size = 100_000
    seed = 1
    # eval
    eval_frequency = 10000
    eval_episodes = 30
    # misc
    log_frequency_step = 10000
    log_save_tb = True
    save_video = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # observation
    from_pixels = True
    image_size = 84
    image_pad = 4
    frame_stack = 3
    feature_dim = 50
    # global params
    # IMPORTANT= please use a batch size of 512 to reproduce the results in the paper. Hovewer, with a smaller batch size it still works well.

    aug = "random_trans"


class Agent(PrefixProto):
    # obs_shape = None
    # action_shape = None
    # action_range = None  # to be specified later
    # device= ${device}
    # encoder_cfg= ${encoder}
    # critic_cfg= ${critic}
    # actor_cfg= ${actor}
    lr = 1e-3
    batch_size = 128
    discount = 0.99
    init_temperature = 0.1
    actor_update_frequency = 2
    critic_tau = 0.01
    critic_target_update_frequency = 2
    # batch_size= ${batch_size}
