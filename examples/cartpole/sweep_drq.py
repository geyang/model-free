from pathlib import Path

from params_proto.hyper import Sweep

from drq.config import Args, Agent
from examples import RUN

with Sweep(RUN, Args, Agent) as sweep:
    Args.from_pixels = False

    with sweep.product:
        with sweep.zip:
            Args.env_name = ['dmc:Cartpole-swingup-v1']
            Args.feature_dim = [50]
            Args.train_frames = [1_000_000]
            Args.replay_buffer_size = [1_000_000]
            Agent.batch_size = [256]
            Agent.lr = [1e-4]
        Args.seed = [100, 200, 300]


sweep.save(f"{Path(__file__).stem}.jsonl")
