AWS_REGIONS = ["ap-northeast-1", "ap-northeast-2", "ap-south-1", "ap-southeast-1", "ap-southeast-2", "eu-central-1",
               "eu-west-1", "sa-east-1", "us-east-1", "us-east-2"]

LAUNCHED_FILE = 'launched.jsonl'

if __name__ == '__main__':
    import os, sys
    from drq.drq import train

    sys.path.remove(os.path.dirname(__file__))  # Guido, please...

    import jaynes
    from ml_logger import logger, USER, ML_Logger
    from params_proto.neo_hyper import Sweep

    from model_free_analysis import RUN, instr, AWS_REGIONS, IMAGE_IDS
    from drq.config import Args

    ledger = ML_Logger(root=os.getcwd())

    # logic for saving and loading the sweep as a file
    sweep = Sweep(Args, RUN).load("all_dmc_state.jsonl")

    # logic for automatically running locally when you are in debug mode
    if 'pydevd' in sys.modules:
        jaynes.config("local")
    else:
        # jaynes.config("visiongpu-docker")
        jaynes.config("supercloud")
    for i, deps in enumerate(sweep):
        thunk = instr(train, deps)
        jaynes.run(thunk)

    jaynes.listen()

    launched = ledger.load_jsonl(LAUNCHED_FILE)
    launched = [i['job_counter'] for i in launched] if launched else None

    for i, deps in enumerate(sweep):
        RUN(deps, script_path=__file__)
        if logger.glob("metrics.pkl", wd=RUN.PREFIX):
            print(i + 1, RUN.PREFIX, "has been ran")
            continue
        if launched and RUN.job_counter in launched:
            print(i + 1, RUN.PREFIX, "has been launched as", RUN.job_counter)
            continue
        print("launching", RUN.job_counter, RUN.PREFIX)
        thunk = instr(train, deps)
        logger.log_text("""
        charts:
        - yKey: train/episode_reward/mean
          xKey: step
        - yKey: eval/episode_reward
          xKey: step
        """, filename=".charts.yml", dedent=True)
        # jaynes.run(thunk)
        while True:
            try:
                jaynes.config("ec2",
                              launch={'region': REGION, 'availability_zone': REGION + 'a',
                                      'image_id': IMAGE_IDS[REGION]['image_id'],
                                      'key_name': USER + '-' + REGION, 'ec2_name': RUN.JOB_NAME[:128]})
                request_id = jaynes.run(thunk)
                ledger.log_metrics(job_counter=RUN.job_counter, request_id=request_id, region=REGION, prefix=RUN.PREFIX,
                                   file=LAUNCHED_FILE, flush=True)
                break
            except Exception as e:
                print(e)
                assert AWS_REGIONS, "can not be empty"
                REGION = AWS_REGIONS.pop(0)
                print("switch to", REGION)

    jaynes.listen()
