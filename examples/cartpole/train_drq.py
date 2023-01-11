if __name__ == '__main__':
    from ml_logger import instr
    from examples import RUN
    import jaynes
    from drq.drq import train
    from drq.config import Args, Agent
    from params_proto.hyper import Sweep

    sweep = Sweep(RUN, Args, Agent).load("sweep_drq.jsonl")
    jaynes.config('supercloud-tg', verbose=True)
    for i, kwargs in enumerate(sweep):
        # RUN.job_name = f"{{now:%H.%M.%S}}/{Args.env_name.split(':')[-1][:-3]}/{Args.seed}"
        # RUN.CUDA_VISIBLE_DEVICES = str(i)
        thunk = instr(train, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
