# Your Goal

- The repo has the following structure that contains an `examples` folder
    
    ```
    model-freeL 1 model-free
    ├── LICENSE
    ├── README.md
    ├── VERSION
    ├── drq
    ├── drq_analysis
    ├── drq_rff
    ├── drqv2
    ├── examples <== You start here. Add a few training scripts (with jaynes)
    ├── ffn_analysis
    ├── ppo
    ├── requirements.txt
    ├── sac_dennis_rff
    └── setup.py
    ```

Your goal is to add a few scripts:

1. one for sac_dennis (no RFF!!) can you make a clean version??
2. one for ppo (not going to work well though, what’s the point?)

Our goal is to move to Isaac gym as soon as possible because the best method for each domain **is now completely different.** It does not really make sense to tweak things on dm_control so that it can work on Isaac Gym. General knowledge though, is useful.