import torch.cuda
from params_proto import ParamsProto, PrefixProto, Proto


class Args(ParamsProto):
    seed = 1

    # task settings
    env_name = 'distracting_control:Cheetah-run-easy-v1'  # distracting env
    # env_name = 'dmc:Cheetah-run-v1'  # non-distracting env
    # todo: use dmc:Quadruped-walk-v1 and gym-dmc module.
    frame_stack = 3
    action_repeat = 2
    discount = 0.99
    # train settings
    train_frames = 1_000_000
    num_seed_frames = 4000
    # eval
    eval_every_frames = 10000
    num_eval_episodes = 10
    # replay buffer
    replay_buffer_size = 1_000_000
    replay_buffer_num_workers = 4
    # todo: remove, replace with frame_stack
    nstep = 3
    batch_size = 256

    # misc
    update_freq = 2
    save_video = True
    log_freq = 100
    checkpoint_freq = 200_000
    checkpoint_root = Proto(env="$ML_LOGGER_BUCKET/checkpoints",
                            help="upload snapshots here at the end, used to "
                                 "resuming after preemption. Local, S3, or Google Storage.")

    # NOTE: If NFS directory is not used, set nfs_dir and local_dir to the same path.
    #   Then file transfer between NFS and local filesystem will not be performed.
    tmp_dir = Proto(None, env="TMPDIR",
                    help="directory for temporarily saving replay buffer and snapshot. All of the "
                         "files will be moved under snapshot_dir after the training finishes")


# todo: merge into Args, pass in explicitly into agent.
class Agent(PrefixProto):
    lr = 1e-4
    critic_target_tau = 0.01
    # use_tb = ${use_tb}
    num_expl_steps = 2000
    hidden_dim = 1024
    feature_dim = 50
    stddev_schedule = 'linear(1.0,0.1,500000)'
    stddev_clip = 0.3
