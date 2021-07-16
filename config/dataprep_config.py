import datetime
import os

"""
defining classes for different setting parameters
for data preprocessing

classes:
--------
    paths
    dataprep_settings
"""

class dataprep_settings:
    """
    Define variables and settings for data preprocessing.
    """
    # ---------------SET MANUALLY---------------

    # DATA SOURCE AND DATA SET CODE
    DATABASE = "WDB"  # stands for Wharton Data Base

class dataprep_paths:
    # ---------------LEAVE---------------

    # data paths
    DATA_PATH = "data"
    RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
    INTERMEDIATE_DATA_PATH = os.path.join(DATA_PATH, "intermediate")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "preprocessed")

    # stock data files, un-preprocessed
    RAW_DATA_FILE_US = os.path.join(RAW_DATA_PATH, "US_stocks_WDB.csv")
    RAW_DATA_FILE_JP = os.path.join(RAW_DATA_PATH, "JP_stocks_WDB.csv")

    # dummy data file (only for testing the algorithm, not real data)
    DUMMYDATA = os.path.join(INTERMEDIATE_DATA_PATH, "dummydata.csv")

