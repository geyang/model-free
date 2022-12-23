from ml_logger import logger

if __name__ == '__main__':
    import os, sys

    sys.path.remove(os.path.dirname(__file__))  # Guido, please...

    from params_proto.neo_hyper import Sweep

    from model_free_analysis import RUN
    from drq.config import Args

    with Sweep(RUN, Args) as sweep:
        # Uncomment these lines to have the data saved to your baseline folder
        RUN.restart = True
        RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
        RUN.job_name = "{job_postfix}"

        # do not save video
        Args.save_video = False

        with sweep.product:
            with sweep.chain:
                with sweep.zip:
                    Args.env_name = [
                        # The PlaNet Benchmark
                        'dmc:Ball_in_cup-catch-v1',
                        'dmc:Cartpole-swingup-v1',
                        'dmc:Cheetah-run-v1',
                        'dmc:Finger-spin-v1',
                        'dmc:Reacher-easy-v1',
                        'dmc:Walker-walk-v1']
                    Args.train_steps = [
                        500_000, 500_000, 2000_000,
                        500_000, 500_000, 1000_000]

                with sweep.zip:
                    Args.env_name = [
                        # The Dreamer Benchmark
                        # 'dmc:Ball_in_cup-catch-v1',
                        'dmc:Cartpole-balance-v1',
                        'dmc:Cartpole-balance_sparse-v1',
                        # 'dmc:Cartpole-swingup-v1',
                        'dmc:Cartpole-swingup_sparse-v1',
                        # 'dmc:Cheetah-run-v1',  # included in PlaNet
                        # 'dmc:Finger-spin-v1',
                        'dmc:Hopper-hop-v1',
                        'dmc:Hopper-stand-v1',
                        'dmc:Pendulum-swingup-v1',
                        # 'dmc:Reacher-easy-v1',
                        'dmc:Reacher-hard-v1',
                        'dmc:Walker-run-v1',
                        'dmc:Walker-stand-v1',
                        # 'dmc:Walker-walk-v1',  # included in PlaNet
                    ]
                    Args.train_steps = [
                        # 500_000,
                        500_000, 500_000, # 1000_000,
                        1000_000,  # 2000_000, 2000_000,
                        2000_000, 2000_000,
                        2000_000, 2000_000,  # 2000_000,
                        2000_000, 1000_000,  # 1000_000
                    ]

            Args.seed = [100, 200, 300, 400, 500]


    def fn(RUN, Args):
        RUN.job_postfix = f"{Args.env_name.split(':')[1][:-3].lower()}/drq/{Args.seed}"
        RUN.job_counter = logger.count('job_counter')


    sweep.each(fn)
    # logic for saving and loading the sweep as a file
    sweep.save(__file__[:-3] + ".jsonl")
