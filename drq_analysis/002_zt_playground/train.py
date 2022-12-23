if __name__ == '__main__':
    import os, sys

    sys.path.remove(os.path.dirname(__file__))  # Guido, please...

    import jaynes
    from ml_logger import logger
    from params_proto.neo_hyper import Sweep

    from model_free_analysis import RUN, instr
    from drq.config import Args

    with Sweep(RUN, Args) as sweep:
        # Uncomment these lines to have the data saved to your baseline folder
        # RUN.restart = True
        # RUN.prefix = "{username}/{project}/{file_stem}/{job_name}"
        # RUN.job_name = "{job_postfix}"

        with sweep.product:
            with sweep.zip:
                Args.env_name = ['dmc:Walker-walk-v1', 'dmc:Walker-run-v1',
                                 'dmc:Hopper-hop-v1', 'dmc:Cheetah-run-v1']
                Args.train_steps = [1000_000, 2000_000, 2000_000, 2000_000]

            Args.seed = [100, 200, 300, 400, 500]


    def fn(RUN, Args):
        RUN.job_postfix = f"{Args.env_name.split(':')[1][:-3].lower()}/drq/{Args.seed}"
        # RUN.job_id = logger.count('job_counter')  # used to distinguish jobs from each other


    sweep.each(fn)
    # logic for saving and loading the sweep as a file
    # sweep.save("sweep.jsonl")
    # sweep = Sweep(Args, RUN).load("sweep.jsonl")

    # logic for automatically running locally when you are in debug mode
    if 'pydevd' in sys.modules:
        jaynes.config("local")
    else:
        jaynes.config("visiongpu-docker")

    from drq.drq import train

    for i, deps in enumerate(sweep[:1]):
        print(i, deps)
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
