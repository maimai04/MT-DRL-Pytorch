# Differences with the original version used
## Original version:
This repository provides codes for [ICAIF 2020 paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)
This ensemble strategy is reimplemented in a Jupiter Notebook at [FinRL](https://github.com/AI4Finance-LLC/FinRL-Library).
Hongyang Yang, Xiao-Yang Liu, Shan Zhong, and Anwar Walid. 2020. Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. In ICAIF ’20: ACM International Conference on AI in Finance, Oct. 15–16, 2020, Manhattan, NY. ACM, New York, NY, USA.

## Original version structure of env folder:
 
```
env/
    EnvMultipleStock_train.py : training environment
    EnvMultipleStock_validation.py: validation environment
    EnvMultipleStock_trade.py: trade (~test) environment
```

## New structure of env folder
same

# Differences summary: old (paper) vs new (my) version
## Common changes to all 3 EnvMultipleStock__.py files
```
- (hyper)parameters were all hard-coded: replaced them with variables and summarized them in the config.py file
- state space construction (line...) was hard-coded: replaced with scalable version
```

## EnvMultipleStock_trade.py
## EnvMultipleStock_validation.py
## EnvMultipleStock_training.py



