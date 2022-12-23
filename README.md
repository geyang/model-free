# Model-free RL Baselines


This codebase combines DrQ, AutoDrac, PPO into a single codebase

 

## Algorithms

This repository contains implementations of the following papers in a unified framework:

- [RAD (Laskin et al., 2020)](https://arxiv.org/abs/2004.14990)
- [CURL (Srinivas et al., 2020)](https://arxiv.org/abs/2004.04136)
- [SAC (Haarnoja et al., 2018)](https://arxiv.org/abs/1812.05905)

using standardized architecture and hyper-parameters, wherever applicable. 

## Setup

All launching and analysis code resides in the `model_free_analysis` sub folder. The `.jaynes` configuration files is under project root. 

Each experiment will occupy one sub folder. ML-Logger will automatically log according to this folder structure. This logic is implemented in [./rl_transfer_analysis/__init__.py](./rl_transfer_analysis/__init__.py).


## ML-Logger Env Setup

1. Install `ml-logger==0.7.11`.

2. Add the following environment variables to your `~/.bashrc`

   ```bash
   export ML_LOGGER_ROOT=http://44.241.150.228:8080
   export ML_LOGGER_USER=$USER
   export ML_LOGGER_TOKEN=
   ```

## Launching via SSH and Docker (on the VisionGPU Cluster)

1. **Update `jaynes`** to `v0.6.15`. This one contains matching updates for this code base.

   ```bash
   pip install jaynes==0.6.10
   ```

2. add `NFS_PATH=/data/pulkitag/misc/$USER` to your `.bashrc` file. We use this parameter in the `.jaynes.config`.

   ```bash
   echo "export NFS_PATH=/data/pulkitag/misc/$USER" >> ~/.bashrc
   ```

3. Install `aws-cli` using the following command:

   ```bash
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   ./aws/install -i $NFS_PATH/aws-cli -b $NFS_PATH/bin
   echo "export PATH=$NFS_PATH/bin:$PATH" >> ~/.bashrc
   ```

   After this, you should be able to run

   ```bash
   aws info
   ```

4. Adding your `aws` credentials, so that you could use the `s3` copy command

   ```bash
   aws configure
   ```

   If `aws s3 cp <bucket-path> <local-path>` is already working, you can skip 2 and 3. 

Now if you run with 

```python
jaynes.configure('visiongpu')
jaynes.run(train_fn)
jaynes.listen()
```

it should correctly package and launch on the vision gpu cluster.

## Training & Evaluation

**This is stale, needs to be updated** The `scripts` directory contains training and evaluation bash scripts for all the included algorithms. Alternatively, you can call the python scripts directly, e.g. for training call

```
python3 src/train.py \
    --algorithm soda \
    --aux_lr 3e-4 \
    --seed 0
```

to run SODA on the default task, `walker_walk`. This should give you an output of the form:

```
Working directory: logs/walker_walk/soda/0
Evaluating: logs/walker_walk/soda/0
| eval | S: 0 | ER: 26.2285 | ERTEST: 25.3730
| train | E: 1 | S: 250 | D: 70.1 s | R: 0.0000 | ALOSS: 0.0000 | CLOSS: 0.0000 | AUXLOSS: 0.0000
```
where `ER` and `ERTEST` corresponds to the average return in the training and test environments, respectively. You can select the test environment used in evaluation with the `--eval_mode` argument, which accepts one of `(train, color_easy, color_hard, video_easy, video_hard)`.


## Results

Work-in-progress


## Acknowledgements

We want to thank the numerous researchers and engineers involved in work of which this implementation is based on. This benchmark is a product of our work on <work-in-progress>, our SAC implementation is based on [this repository](https://github.com/denisyarats/pytorch_sac_ae), the original DMControl is available [here](https://github.com/deepmind/dm_control), and the gym wrapper for it is available [here](https://github.com/denisyarats/dmc2gym). RAD, and CURL baselines are based on their official implementations provided [here](https://github.com/nicklashansen/policy-adaptation-during-deployment), [here](https://github.com/MishaLaskin/rad), and [here](https://github.com/MishaLaskin/curl), respectively.
