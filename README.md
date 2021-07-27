MAIN README FILE


## files in this folder
```
run.py

requirements.txt
```
## INFO
There are readme files in every folder and there are commments in the code.

## RUN THE PROJECT
The workflow is as follows:
1) set the configurations in the config.py file in the config folder
2) run the setup => type: python run.py in the console

## PROJECT STRUCTURE

```
MT_BarbaraCapl
    analysis        : stores results folders with plots and ipython notebooks for analysis
    config          : stores configurations for data preprocessing (dataprep_config) and for the run (config.py)
    data            : stores the data
    dataprep        : stores ipython files for data exploration and preparation
    environment     : stires class for the environment
    model           : stores classes for PPO agent, Buffer and actor critic networks
    pipeline        : implements the train / test / validation loop for the rolling window approach
    results         : stores result files of each run
    trained models  : stores trained models of each run

run.py              : runs the whole setup
requirements.txt    : stores requirements
```
