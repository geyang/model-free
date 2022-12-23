import os

import matplotlib.pyplot as plt
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
# DrQ Learning Curves

```
"""
with doc:
    colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247', "brown", "gray"]
    loader = ML_Logger(prefix="model-free/model-free/001_baselines/train")

with doc.hide:
    def plot_line(path, color, label, show_train=True, linestyle=None):
        if show_train:
            median, step, = loader.read_metrics("train/episode_reward/mean@mean",
                                                x_key="step@min", path=path, bin_size=5, )
            plt.plot(step.to_list(), median.to_list(), color=color, linestyle="--")

        mean, top, bottom, step, = loader.read_metrics("eval/episode_reward@mean",
                                                       "eval/episode_reward@68%",
                                                       "eval/episode_reward@33%",
                                                       x_key="step@min", path=path)
        doc.print(path, step.max())
        plt.plot(step.to_list(), mean.to_list(), linestyle=linestyle, color=color, label=label)
        plt.fill_between(step, bottom, top, alpha=0.15, color=color)

doc @ """
Plotting the learning curve at various regularization strength
"""

with doc.hide, doc.table() as table:
    for i, env_name in enumerate(tqdm(all_games)):
        r = table.figure_row() if i % 4 == 0 else r

        with r:
            plt.figure(figsize=(3, 2))

            plot_line(path=f"{env_name}/drq/**/metrics.pkl", color=colors[0], label=f"DrQ")
            plot_line(path=f"{env_name}/drq-state/**/metrics.pkl", color=colors[1], label=f"SAC(sate)")

            plt.tight_layout()
            r.savefig(f"{os.path.basename(__file__)[:-3]}/{env_name}.png", title=env_name, dpi=300, bbox_inches='tight')

            if i == len(all_games) - 1:
                plt.legend(frameon=False)
                ax = plt.gca()
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.collections.clear()
                ax.lines.clear()
                [s.set_visible(False) for s in ax.spines.values()]
                r.savefig(f"{os.path.basename(__file__)[:-3]}/legend.png", title=env_name, dpi=300,
                          bbox_inches='tight', pad_inches=0)
            plt.close()

doc.flush()
