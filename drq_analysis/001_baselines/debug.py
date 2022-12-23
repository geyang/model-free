if __name__ == '__main__':
    import os, sys
    from drq.drq import train

    sys.path.remove(os.path.dirname(__file__))  # Guido, please...

    import jaynes
    from ml_logger import logger
    from params_proto.neo_hyper import Sweep

    from model_free_analysis import RUN, instr
    from drq.config import Args

    with Sweep(RUN, Args) as sweep:
        # Uncomment these lines to have the data saved to your baseline folder
        RUN.restart = True
        RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
        RUN.job_name = "{job_postfix}"

        # do not save video
        Args.save_video = False
        Args.from_pixels = False

        with sweep.product:
            with sweep.chain:
                with sweep.zip:
                    Args.env_name = [
                        'dmc:Cartpole-balance-v1',
                        'dmc:Cartpole-balance_sparse-v1',
                        'dmc:Cartpole-swingup_sparse-v1',
                    ]
                    Args.train_steps = [500_000, 500_000, 1000_000, ]

            Args.seed = [100]  # , 200, 300, 400, 500]


    @sweep.each
    def fn(RUN, Args):
        RUN.job_postfix = f"{Args.env_name.split(':')[1][:-3].lower()}/drq-state/{Args.seed}"
        RUN.job_counter = logger.count('job_counter')


    # logic for automatically running locally when you are in debug mode
    if 'pydevd' in sys.modules:
        jaynes.config("local")
    else:
        # jaynes.config("visiongpu-docker")
        jaynes.config("supercloud")
    for i, deps in enumerate(sweep):
        thunk = instr(train, deps)
        logger.log_text("""
        charts:
        - yKey: train/episode_reward/mean
          xKey: step
        - yKey: eval/episode_reward
          xKey: step
        """, filename=".charts.yml", dedent=True)
        jaynes.run(thunk)

    jaynes.listen()
