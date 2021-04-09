import datetime
import os

"""
defining classes for grouping parameters.

classes:
--------
    settings
    crisis_settings
    paths
    env_params
"""

class settings:
    """
    Define general settings for the whole run and global variables.
    """
    # ---------------SET MANUALLY---------------
    ### strategy mode to be run
    STRATEGY_MODE = "ppo"
    # STRATEGY_MODE = "ensemble"
    #STRATEGY_MODE = None # only used for debugging (no models are run then)

    ### set random seeds;
    SEED_PPO = 223445
    SEED_ENV = 101882

    # ---------------LEAVE---------------
    ### Set dates
    STARTDATE_TRAIN = 20090000 # Note: this is also the "global startdate"
    ENDDATE_TRAIN = 20150000

    STARTDATE_VALIDATION = 20151001
    ENDDATE_VALIDATION = 20200707

    ### set windows
    # REBALANCE_WINDOW is the number of months to retrain the model
    # VALIDATION_WINDOW is the number of months to validate the model and select the DRL agent for trading
    # 63 days = 3 months of each 21 trading days (common exchanges don't trade on weekends, need to change for crypto)
    REBALANCE_WINDOW = 63 # this is basically a "step" period; we take a step of 63 days for which we extend the training window and move the validation and trade window
    # todo: rename to STEP_WINDOW
    VALIDATION_WINDOW = 63

    ### returns current timestamp, mainly used for naming directories/ printout / logging to .txt
    NOW = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

class env_params:
    # ---------------SET MANUALLY---------------

    # ---------------LEAVE---------------
    HMAX_NORMALIZE = 100 # todo: up and check perf
    INITIAL_CASH_BALANCE = 1000000
    TRANSACTION_FEE_PERCENT = 0.001 # todo: check reasonab
    REWARD_SCALING = 1e-4 # todo: check
    # shorting possiblity? # todo

class ppo_params:
    # ---------------SET MANUALLY---------------

    # ---------------LEAVE---------------
    TRAINING_TIMESTEPS = 100000


class crisis_settings:
    """
    Choose if you want to use a crisis measure (such as turbulence and other) to act as a stop loss in times of
    turbulence.
    Choose parameters and settings for your crisis measure of choice.
    """

    # ---------------SET MANUALLY---------------
    #CRISIS_MEASURE = None  # default
    CRISIS_MEASURE = "turbulence"
    # CRISIS_MEASURE = "volatility"

    # ---------------LEAVE---------------
    if CRISIS_MEASURE == "turbulence":
        CNAME = "turb"
        #CRISIS_THRESHOLD = 140  # turbulence threshold
        #CRISIS_DATA = None  # path to pre-calculated data
        CUTOFF_XPERCENTILE = .90
        print("crisis condition measure: {}".format(CRISIS_MEASURE))
    elif CRISIS_MEASURE == "volatility":
        CNAME = "vola"
        CUTOFF_XPERCENTILE = ""
        #CRISIS_THRESHOLD = None  # turbulence threshold
        #CRISIS_DATA = None
        print("crisis condition measure: {}".format(CRISIS_MEASURE))
    elif CRISIS_MEASURE is None:
        CNAME = ""
        CUTOFF_XPERCENTILE = ""
        print("no crisis measure selected.")
    else:
        print("ValueError: crisis measure selected unkNOWn and not None.")

class paths:
    # ---------------LEAVE---------------

    # data paths
    DATA_PATH = "data"
    RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "preprocessed") # todo: was PREPROCESSED_RAW_DATA_PATH

    # trained models and results path
    TRAINED_MODELS_PATH = "trained_models"
    RESULTS_PATH = "results"
    # names of sub-directories within results folder (based on the memories we save in the env.)
    SUBDIR_NAMES = {"cash_value": "cash_value",
                    "portfolio_value": "portfolio_value",
                    "rewards": "rewards",
                    "policy_actions": "policy_actions",
                    "exercised_actions": "exercised_actions",
                    "transaction_cost": "transaction_cost",
                    "number_asset_holdings": "number_asset_holdings",
                    "sell_trades": "sell_trades",
                    "buy_trades": "buy_trades",
                    "crisis_measures": "crisis_measures",
                    "crisis_thresholds": "crisis_thresholds",
                    "state_memory": "state_memory"}

    # data files
    TESTING_DATA_FILE = "test.csv"  # TODO: WHERE IS THIS FILE? rm
    RAW_DATA_FILE = os.path.join(RAW_DATA_PATH, "dow_30_2009_2020.csv") # todo: was TRAINING_DATA_FILE
    PREPROCESSED_DATA_FILE = os.path.join(PREPROCESSED_DATA_PATH, "done_data.csv") # todo: was PREP_DATA_FILE

class dataprep_settings:
    """
    Define variables and settings for data preprocessing.
    """
    # ---------------SET MANUALLY---------------
    ### choose if you want to use pre-processed data (False) or raw data and pre-process (True)
    PREPROCESS_ANEW = False  # default
    #PREPROCESS_ANEW = True

    ### CHOOSE WHICH ARE THE (MANDATORY) BASE COLUMNS; for Wharton DB: datadate, tic
    BASE_DF_COLS = ['datadate', 'tic']
    ASSET_NAME_COLUMN = "tic"
    DATE_COLUMN = "datadate"
    MAIN_PRICE_COLUMN = "adjcp"

    ### CHOOSE SUBSET OF COLUMNS TO BE LOADED IN FROM RAW DATAFRAME
    # depends on the dataset used (by default: Wharton Database)
    # needs to be tailored depending on dataset / data source
    RAW_DF_COLS_SUBSET = BASE_DF_COLS + ['prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']

    ### CHOOSE WHICH NEW COLUMNS SHOULD BE CREATED BASED ON RAW_DF_COLS_SUBSET
    # by default: using Wharton DB data
    NEW_COLS_SUBSET = ['adjcp', 'open', 'high', 'low', 'volume']


    ### PROVIDE NAMES OF ALL FEATURES / INDICATORS GIVEN DATASET COLUMN NAMES # todo
    TECH_INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"] # was FEATURES_LIST
    RISK_INDICATORS = ["vola"]
    ESG_INDICATORS = ["ESG"]
    LSTM_INDICATORS = []

    ### CHOOSE WHICH COLUMNS YOU WANT TO USE IN YOUR FINAL, CLEAN DATASET
    ALL_USED_COLUMNS = BASE_DF_COLS + ['adjcp'] + TECH_INDICATORS

    # list of all features to be used apart from (here: daily closing) price
    FEATURES_LIST = ["macd", "rsi", "cci", "adx"] # todo
    # todo

    #PREPROCESSING_LEVEL = "prepLevel01"
    #FEATURES_LIST = TECH_INDICATORS + RISK_INDICATORS + ESG_INDICATORS + LSTM_INDICATORS
    ### choose level of pre-processing / data preparation
    #PREPROCESSING_LEVEL = "prepLevel00" # using only adjusted closing prices (adjcp)
    #PREPROCESSING_LEVEL = "prepLevel01" # using adjcp + technical indicators (as specified under TECH_INDICATORS)
    #PREPROCESSING_LEVEL = "prepLevel02" # using adjcp + technical indicators + volatility
    #PREPROCESSING_LEVEL = "prepLevel03" # using adjcp + technical indicators + volatility + LSTM
    #PREPROCESSING_LEVEL = "prepLevel04" # using adjcp + technical indicators + volatility + LSTM + ESG
    #PREPROCESSING_LEVEL = "prepLevel05" # using adjcp + technical indicators + volatility + ESG
    #PREPROCESSING_LEVEL = "prepLevel06" # using adjcp + volatility
    #PREPROCESSING_LEVEL = "prepLevel07" # using adjcp + volatility + LSTM
    #PREPROCESSING_LEVEL = "prepLevel07" # using adjcp + volatility + LSTM + ESG
    #PREPROCESSING_LEVEL = "prepLevel07" # using adjcp + volatility + ESG

    # ---------------LEAVE---------------

    if FEATURES_LIST == ["macd", "rsi", "cci", "adx"]: # todo
        FEATURES_MODE = "fm1"
    if FEATURES_LIST == ["adjcp"]: # todo
        FEATURES_MODE = "fm2"
