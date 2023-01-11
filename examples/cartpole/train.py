if __name__ == '__main__':
    from ml_logger import instr
    from examples import RUN
    import jaynes
    from sac_dennis_rff.sac import train
    from sac_dennis_rff.config import Args, Actor, Critic, Agent
    from params_proto.hyper import Sweep

    sweep = Sweep(RUN, Args, Actor, Critic, Agent).load("sweep.jsonl")
    jaynes.config('supercloud-tg', verbose=True)
    for i, kwargs in enumerate(sweep):
        # RUN.job_name = f"{{now:%H.%M.%S}}/{Args.env_name.split(':')[-1][:-3]}/{Args.seed}"
        # RUN.CUDA_VISIBLE_DEVICES = str(i)
        thunk = instr(train, **kwargs)
        jaynes.run(thunk)

    jaynes.listen()
