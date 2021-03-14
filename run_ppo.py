# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *
import os

"""
run only PPO
everything else c.p.

additional changes from BCAP:
- added comments
- added variable which_run, to save subsequent runs in corresponding folders.
- 
"""
# BCAP: do you want to pre-process the data again and overwrite the already-preprocessed data currently saved
# under done_data.csv?
preprocess_anew = False

def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "done_data.csv"

    # BCAP: added second condition so we can choose to re-preprocess data
    if os.path.exists(preprocessed_path) or preprocess_anew==False:
        # BCAP: added comment
        print("Using existing pre-processed data under path done_data.csv.")
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        # BCAP: added comment
        print("Preprocessing data.")
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    ## PPO Only
    run_single_agent(df=data,
                     unique_trade_date=unique_trade_date,
                     rebalance_window=rebalance_window,
                     validation_window=validation_window, strategy_name="PPO")

    # _logger.info(f"saving model version: {_version}")


if __name__ == "__main__":
    run_model()
