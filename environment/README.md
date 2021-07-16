# Folder: environment

## Overview
This folder contains the following python (.py) file:

```
FinancialMarketEnv.py           : called from run_pipeline.py
    class FinancialMarketEnv    : subclass of gym.Env (OpenAI gym Environment), which doesn't mean much just that 
                                  the environment must meet some basic gym standards (e.g. it should have implemented
                                  certain methods such as .reset(, .step() etc.)
                                  It takes as input certain parameters + the data (train, validation or test) and
                                  can be then queried for the current state or next state (using actions).    
```