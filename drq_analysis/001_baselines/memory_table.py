from cmx import doc
from ml_logger import ML_Logger
from tqdm import tqdm

all_games = """
ball_in_cup-catch
cartpole-swingup
cheetah-run
finger-spin
reacher-easy
walker-walk
cartpole-balance
cartpole-balance_sparse
cartpole-swingup_sparse
hopper-hop
hopper-stand
pendulum-swingup
reacher-hard
walker-run
walker-stand
""".strip().split('\n')

doc @ """
# Running speed over various environments

The different envs are running at different speeds makes me wonder if there is not enough memory.
"""
colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247', "brown", "gray"]
loader = ML_Logger(prefix="model-free/model-free/001_baselines/train")

def mem_dt(path):
    env_name = loader.read_params("Args.env_name", path=path.replace('metrics.pkl', 'parameters.pkl'))
    dt, step = loader.read_metrics("dt_episode@mean", x_key="step@min", path=path)
    return env_name, dt.mean(), step.max()


with doc:
    import pandas as pd

    all = []
    for i, env_name in enumerate(tqdm(all_games)):
        env_name, dt, step = mem_dt(path=f"{env_name}/drq-state/**/metrics.pkl")
        all.append({'env_name': env_name, 'dt': dt, 'last_step': step, 'prod': dt * step})

    table = pd.DataFrame(all).round(1)
    doc.csv @ table
