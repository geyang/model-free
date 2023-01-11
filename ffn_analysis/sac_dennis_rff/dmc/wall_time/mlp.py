from pathlib import Path

from params_proto.hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from ffn_analysis import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.checkpoint_root = "gs://your-gs-bucket"
    Args.save_final_replay_buffer = True

    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"

    Actor.hidden_layers = 3
    Critic.hidden_layers = 3

    Args.train_frames = 1_000_000

    with sweep.product:
        Args.seed = [100, 200, 300, 400, 500]
        with sweep.zip:
            Args.env_name = ['dmc:Quadruped-run-v1',
                             'dmc:Walker-run-v1']


@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"{Args.env_name.split(':')[-1][:-3]}/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")
