from cmx import doc

doc @ """
# Get All Environment Prefixes
"""

with doc:
    from ml_logger import logger

    with logger.Prefix("/model-free/model-free/001_baselines/train"):
        all_envs = logger.glob('*')
        doc.print(*all_envs, sep="\n")


doc.flush()