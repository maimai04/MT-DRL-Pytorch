# Folder: pipeline

## Overview
This folder contains the following python (.py) files (listed in the order they are called from the main run.py file),  
which contain the following functions:

```
setup_functions.py                : Called from run.py. Used to set up the run before run_pipeline.py is called.
    load_dataset()                : loads the data set from the data diretcory as pandas dataframe
    create_dirs()                 : creates results directories for the current run
    config_logging_to_txt         : saves all the configurations / agent and environment parameters
                                    from config.py in a txt file (configurations.txt) into the folder _LOGGINGS
                                    for each tun (found under results / nameofthecurrentrun / _LOGGINGS) 
    get_data_params               : gets from loaded dataframe the number of assets and number of features so we can 
                                    later pass it to other functions that need this info (e.g. the environment)

run_pipeline.py                   : Called from run.py. Runs the whole model setup 
    run_expanding_window_setup()  : runs the whole pipeline for the expanding window cross validation,
                                    then does some backtesting,
                                    and creates plots for the most important results.

support_functions.py              : Called from run_pipeline.py.
    get_model()                   : based on our configurations, it gets the specified RL agent; 
                                    imports the models from model folder, if we are using the custom implementation,
                                    or imports the ppo agent from stable baselines 3.
```

## Expanding window setup 
# todo
