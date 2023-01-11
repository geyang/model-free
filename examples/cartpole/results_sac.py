import os

import matplotlib.pyplot as plt
from cmx import doc
from ml_logger import ML_Logger, memoize

if __name__ == "__main__":

    with doc @ """# Analysis example (SAC, on dmc:Cartpole-swingup-v1)""":
        loader = ML_Logger(prefix="/model-free-examples/model-free-examples/")

    colors = ['#ff7575', '#23aaff', '#66c56c', '#f4b247']

    with doc:
        def plot_line(path, color, label, x_key, y_key, linestyle='-'):
            mean, low, high, step, = loader.read_metrics(f"{y_key}@mean",
                                                         f"{y_key}@16%",
                                                         f"{y_key}@84%",
                                                         x_key=f"{x_key}@min", path=path, dropna=True)
            plt.xlabel('Frames', fontsize=18)
            plt.ylabel('Rewards', fontsize=18)

            plt.plot(step.to_list(), mean.to_list(), color=color, label=label, linestyle=linestyle)
            plt.fill_between(step, low, high, alpha=0.1, color=color)

    with doc:
        r = doc.table().figure_row()

        user = 'timur'
        date = '2023/01-11'
        env_short = 'cartpole'
        env = 'Cartpole-swingup-v1'
        script_name = 'train'
        job_time = '10.43.09'

        plot_line(path=f"{user}/{date}/{env_short}/{script_name}/{job_time}/**/metrics.pkl", color='black', label='SAC',
                  x_key='frames', y_key="eval/episode_reward/mean")

        plt.title(env)
        plt.legend()
        plt.tight_layout()
        [line.set_zorder(100) for line in plt.gca().lines]
        [spine.set_zorder(100) for spine in plt.gca().collections]
        r.savefig(f'{os.path.basename(__file__)[:-3]}/{env}.png', dpi=300, zoom=0.3, title=env)
        plt.savefig(f'{os.path.basename(__file__)[:-3]}/{env}.pdf', dpi=300, zoom=0.3)
        plt.close()

    doc.flush()
