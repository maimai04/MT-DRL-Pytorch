# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import paths, crisis_settings, settings

# model
# TODO: rm unneeded
from model.models import *
import os

"""
run original code from paper GitHub Repo
ensemble deep reinforcement learning
"""
def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "data/preprocessed/done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0) # index_col=0 prevents an index col like "Unnamed: 0" to be created
    else:
        data = preprocess_data()
        data = add_crisis_measure(data)
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
    
    ## Ensemble Strategy
    run_ensemble_strategy(df=data, 
                          unique_trade_date=unique_trade_date,
                          rebalance_window=rebalance_window,
                          validation_window=validation_window)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
