
# Get All Environment Prefixes

```python
from ml_logger import logger

with logger.Prefix("/model-free/model-free/001_baselines/train"):
    all_envs = logger.glob('*')
    doc.print(*all_envs, sep="\n")
    logger.remove('parameters.pkl')
    logger.remove('outputs.log')
```

```
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
```
