from pathlib import Path

from params_proto.hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from examples import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.checkpoint_root = None  # "gs://ge-data-improbable/checkpoints"
    Args.save_final_replay_buffer = True

    Actor.hidden_layers = 2
    Critic.hidden_layers = 2

    with sweep.product:
        Args.seed = [100, 200, 300]
        with sweep.zip:
            Args.env_name = ['dmc:Cartpole-swingup-v1']
            Args.train_frames = [1_000_000]

sweep.save(f"{Path(__file__).stem}.jsonl")
