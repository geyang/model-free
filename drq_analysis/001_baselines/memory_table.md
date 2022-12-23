
# Running speed over various environments

The different envs are running at different speeds makes me wonder if there is not enough memory.

```python
import pandas as pd

all = []
for i, env_name in enumerate(tqdm(all_games)):
    env_name, dt, step = mem_dt(path=f"{env_name}/drq-state/**/metrics.pkl")
    all.append({'env_name': env_name, 'dt': dt, 'last_step': step, 'prod': dt * step})

table = pd.DataFrame(all).round(1)
doc.csv @ table
```

| env_name                       |     dt |   last_step |        prod |
|--------------------------------|--------|-------------|-------------|
| dmc:Ball_in_cup-catch-v1       |   54.5 |      168000 | 9.16286e+06 |
| dmc:Cartpole-swingup-v1        |   70   |      105000 | 7.34545e+06 |
| dmc:Cheetah-run-v1             |  134.6 |      105500 | 1.41997e+07 |
| dmc:Finger-spin-v1             |  784.6 |       17750 | 1.39275e+07 |
| dmc:Reacher-easy-v1            |  566.8 |      102250 | 5.79544e+07 |
| dmc:Walker-walk-v1             |  324.1 |      113750 | 3.68624e+07 |
| dmc:Cartpole-balance-v1        |  726.8 |       15250 | 1.10832e+07 |
| dmc:Cartpole-balance_sparse-v1 |  864.9 |       29250 | 2.52986e+07 |
| dmc:Cartpole-swingup_sparse-v1 | 1052.8 |       10250 | 1.07914e+07 |
| dmc:Hopper-hop-v1              |   57.9 |      109750 | 6.3511e+06  |
| dmc:Hopper-stand-v1            |   23.5 |      119000 | 2.79915e+06 |
| dmc:Pendulum-swingup-v1        |  290.6 |       84250 | 2.44813e+07 |
| dmc:Reacher-hard-v1            |   37.3 |      100500 | 3.74728e+06 |
| dmc:Walker-run-v1              |   41.2 |      189000 | 7.77838e+06 |
| dmc:Walker-stand-v1            |   36.5 |      109500 | 4.00217e+06 |
