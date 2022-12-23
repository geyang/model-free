import torch

from params_proto.neo_proto import ParamsProto, PrefixProto, Proto


class Args(PrefixProto):
    env_name = "Ant-v2"
    dmc = False
    action_repeat = 1
    # train
    train_frames = 1_000_000
    seed_frames = 25_000
    optim_iters = 1
    replay_buffer_size = 1000000
    seed = 1
    # eval
    eval_frequency = 10000
    eval_episodes = 30
    # misc
    log_frequency_step = 10000
    log_save_tb = True
    checkpoint_freq = 30000
    save_video = False
    save_final_replay_buffer = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Normalization constants
    normalize_obs = False
    obs_bias = None
    obs_scale = None
    # observation
    from_pixels = False
    image_size = 84
    image_pad = 4
    frame_stack = 3

    report_rank = False

    aug = "random_trans"
    checkpoint_root = Proto(env="$ML_LOGGER_BUCKET/checkpoints")


class Encoder(PrefixProto):
    dummy = True
    # Not used by dummy encoder
    hidden_layers = 2
    hidden_features = 400
    out_features = 50
    use_dense = Proto(False, help="When true, use dense RL nets")


class Actor(PrefixProto):
    hidden_layers = 2
    hidden_features = 400
    use_dense = Proto(False, help="When true, use dense RL nets")


class Critic(PrefixProto):
    hidden_layers = 2
    hidden_features = 400
    use_dense = Proto(False, help="When true, use dense RL nets")


class Agent(PrefixProto):
    lr = 1e-4
    batch_size = Proto(256, help="please use a batch size of 512 to reproduce the results in the paper. "
                                 "However, with a smaller batch size it still works well.")
    discount = 0.99
    init_temperature = 0.1
    actor_update_frequency = 2
    critic_tau = 0.01
    critic_target_update_frequency = 2

    share_encoder = Proto(True, help="When true, actor and critic share encoder.")
    # RFF constants
    rff_type = 'rff'
    rff_init = 'iso'
    use_rff = Proto(False, help="When true, uses the RFF on the action input")
    in_scale = None
    state_fourier_features = 400
    action_fourier_features = 400
