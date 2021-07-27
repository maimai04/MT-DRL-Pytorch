import datetime
import os
import torch

"""
This config file defines parameters which are used for running the code.
The parameters are grouped into classes.

Note: 
- parameters after ---------------SET MANUALLY--------------- can be changed
- parameters after ---------------DEFAULT--------------- can be changed but are considered default parameters
- parameters after ---------------LEAVE--------------- are created automatically and should not be changed


classes:
--------
    settings
    paths
    env_params
"""

class settings:
    """
    Defining general settings for the whole run and global variables.
    """
    # ---------------SET MANUALLY---------------

    #  ---------------DEFAULT---------------
    # dataset used:
    DATASET = "US_stocks_WDB_full"
    ### strategy mode to be run
    STRATEGY_MODE = "ppoCustomBase"

    REWARD_MEASURE = "addPFVal" # additional portfolio value, = change in portfolio value as a reward
    #REWARD_MEASURE = "logU" # log utility of new / old value, in oder to "smooth out" larger rewards
    #REWARD_MEASURE = "semvarPenalty" # log utility with semivariance penalty

    RETRAIN_DATA = False # = saving trained agent after each run and continue training only on the next train data chunk, using pre-trained agent (faster)
    #RETRAIN_DATA = True # = when training again on the whole training dataset for each episode

    ### Set dates
    # train
    STARTDATE_TRAIN = 20090101 #20141001 #20090102  # Note: this is also the "global startdate"
    ENDDATE_TRAIN = 20151001
    # backtesting
    STARTDATE_BACKTESTING_BULL = 20070605
    ENDDATE_BACKTESTING_BULL = 20070904 # there is no 2./3. sept
    STARTDATE_BACKTESTING_BEAR = 20070904
    ENDDATE_BACKTESTING_BEAR = 20071203

    ### set rollover window; since we are doing rolling window / extended window cross validation for time series
    # 63 days = 3 months of each 21 trading days (common exchanges don't trade on weekends, need to change for crypto)
    ROLL_WINDOW = 63
    VALIDATION_WINDOW = 63
    TESTING_WINDOW = 63

    # ---------------LEAVE---------------
    ### define 10 randomly picked numbers to be used for seeding
    SEEDS_LIST = [0, 5, 23, 7774, 11112]#,  45252, 80923, 223445, 444110]
    SEED = None # placeholder, will be overwritten in run file)

    ### returns current timestamp, mainly used for naming directories/ printout / logging to .txt
    NOW = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    # this is going to be in the run folder name
    if RETRAIN_DATA:
        # if we retrain data, run models "long" (simply because it takes longer)
        RUN_MODE = "lng" # for "long"
    else:
        # if we do not retrain data, run mode is short (the run takes less long)
        RUN_MODE = "st" # for "short"

class data_settings:
    """
    Define settings for datapreparation before the run, such as
    reading out the number of assets in a data set and the feature columns..
    """
    #  ---------------DEFAULT---------------
    # DATA SOURCE AND DATA SET CODE
    DATABASE = "WDB"  # stands for Wharton Data Base
    COUNTRY = "US"

    ### CHOOSE WHICH ARE THE (MANDATORY) BASE COLUMNS; for Wharton DB: datadate, tic
    if DATABASE == "WDB":
        # adjcp (adjusted closing price) is a default column we need in the state space because we need it to calculate
        # the number of stocks we can buy with our limited budget
        MAIN_PRICE_COLUMN = "adjcp"
        # this is the column where we store the tickers, we do not need them in our state space but in order to
        # reformat the data set from long to wide format
        ASSET_NAME_COLUMN = "tic"
        # this is the date column, we need it to split the data set in different train / validation / test sets
        DATE_COLUMN = "datadate"
        # these are the columns which are not used for state representation
        BASE_DF_COLS = [DATE_COLUMN, ASSET_NAME_COLUMN]

    ### FOR DATA PREPROCESSING:
    # 1) Choose subset of columns to be loaded in from raw dataframe
    # depends on the dataset used (by default: Wharton Database)
    # needs to be tailored depending on dataset / data source
    RAW_DF_COLS_SUBSET = BASE_DF_COLS + ['prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']
    # 2) Choose which new columns should be created as intermediate step based on RAW_DF_COLS_SUBSET
    # by default: using Wharton DB data
    NEW_COLS_SUBSET = ['adjcp', 'open', 'high', 'low', 'volume']

    ### PROVIDE NAMES OF ALL FEATURES / INDICATORS GIVEN DATASET COLUMN NAMES
    PRICE_FEATURES = [MAIN_PRICE_COLUMN]
    TECH_INDICATORS = ["macd", "rsi_21", "cci_21", "dx_21"]#, "obv"] # technical indicators for momentum, obv instead of raw "volume"
    RETURNS_FEATURES = ["log_return_daily"] # log returns because they are a bit less "extreme" when they are larger and since we have daily returns this could be practical
    RISK_INDICATORS = ["ret_vola_21d"] # 21 days volatility and daily vix (divide by 100)
    SINGLE_FEATURES = ["vixDiv100"] # not attached to a certain asset

    # only applied if lstm net arch chosen
    LSTM_FEATURES = RETURNS_FEATURES + RISK_INDICATORS + SINGLE_FEATURES


    # ---------------SET MANUALLY---------------
    # CHOOSE FEATURES MODE, BASED ON WHICH THE FEATURES LIST IS CREATED (SEE BELOW)
    FEATURES_MODE = "fm2"
    #FEATURES_MODE = "fm3"
    #FEATURES_MODE = "fm7" # for lstm

    # ---------------LEAVE---------------
    if FEATURES_MODE == "fm1":
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES
        SINGLE_FEATURES_LIST = []
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm2": # features version of the ensemble paper
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS #+ RETURNS_FEATURES
        SINGLE_FEATURES_LIST = []
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm3":
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES + TECH_INDICATORS + RISK_INDICATORS
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm4":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS + RISK_INDICATORS# + RETURNS_FEATURES
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm5":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS + RETURNS_FEATURES
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm6":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm7": # this fetaure mode is the one where we use lstm for return, vola and vis
        # and CUT AWAY these fetaures from the mlp in the Brain
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS + RISK_INDICATORS + RETURNS_FEATURES
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = RETURNS_FEATURES + RISK_INDICATORS + SINGLE_FEATURES
    else:
        print("error (config): features list not found, cannot assign features mode.")

class env_params:
    """
    Parameters used within the environment:

    STEP_VERSION    : indicates the way the actions are exercised in the env when the agent takes a step in the env.
                      This is depending on whether we have actions from a Gaussian distribution, which are "number of assets to buy"
                      or from a Dirichlet distribution, which are target weights.
                      "paper": the version of the paper for gaussian policy is used, where action = number of assets to buy, no short-selling either
                      "newNoShort": the version with the Dirichlet distribution, no short-selling
                      "newNoShort2: version whith Dirichlet where the cash weight is also estimated by the policy network
    """
    # ---------------SET MANUALLY---------------
    STEP_VERSION = "paper"
    #STEP_VERSION = "newNoShort"
    #STEP_VERSION = "newNoShort2"

    # ---------------DEFAULT---------------
    if STEP_VERSION == "newNoShort" or STEP_VERSION == "newNoShort2":
        HMAX_NORMALIZE = None  # This is the max. number of stocks one is allowed to buy of each stock. It is none here since we don't
                               # actually use this parameter in this version, so it could also be any number really
        REWARD_SCALING = 1e-4  # This is 0.0001. It is the number the reward is multiplied with in order to make it smaller, as networks work better with numbers around 0
                               # instead with numbers in the thousands
        REBALANCE_PENALTY = 0  #0.2 # if 0, no penalty, if 1, so much penalty that no change in weight

    elif STEP_VERSION == "paper":
        HMAX_NORMALIZE = 100   # This is the max. number of stocks one is allowed to buy of each stock
        REWARD_SCALING = 1e-4  # This is 0.0001. It is the number the reward is multiplied with in order to make it smaller, as networks work better with numbers around 0
        REBALANCE_PENALTY = None # here not applicable, so None as fill-in value but could be anything

    # starting cash value 1 mio.
    INITIAL_CASH_BALANCE = 1000000
    # transaction fee applied to trading volume
    TRANSACTION_FEE_PERCENT = 0.001  # reasonability: https://www.google.com/search?client=firefox-b-d&q=transaction+fee+for+stock+trading

class agent_params:
    """
    Here, the (hyper-)parameters for the PPO agent / algorithm are defined.
    Initially, there were two ppo versions, that is why there is a class in a class here.
    """
    class ppoCustomBase:
        """
        Here, the hyper-parameters for the custom ppo agent are defined.
        """
        # ---------------SET MANUALLY---------------
        ### SETUP PARAMETERS
        # net architecture mode

        NET_ARCH = "mlp_shared" # only feed-forward network, with fully shared layers between actor and critic
        #NET_ARCH = "mlplstm_shared" # feed-forward network + lstm, with fully shared layers between actor and critic

        # ---------------DEFAULT---------------
        ### HYPERPARAMETERS
        BATCH_SIZE = 64 # = minibatch size, for mini-batch updates during training
        NUM_EPOCHS = 10 # how many times the policy is updated on the whole same batch of data (not minibatch, ut whole batch)
        OPTIMIZER = torch.optim.Adam # optimizer for gradient descent
        OPTIMIZER_LEARNING_RATE = 0.00025 #learning rate for optimizer
        GAMMA = 0.99 # discount factor for rewards
        GAE_LAMBDA = 0.95 # smoothing factor for GAE (generalized advantage estimator) calculation
        CLIP_EPSILON = 0.2 # clip rate for surrogate objective
        CRITIC_LOSS_COEF = 0.5 # how much weight the value loss has in the combined loss

        # setting the entropy coefficient: how much do we value exploration in the combined loss function
        if env_params.STEP_VERSION == "newNoShort" or env_params.STEP_VERSION == "newNoShort2":
            # for version with Dirichlet distribution, higher entropy is better,
            ENTROPY_LOSS_COEF = 1000 #0.0001 #0.005 #0.01 #0.01
        elif env_params.STEP_VERSION == "paper":
            # for version with Gaussian distribution, lower entropy works better, because the standard deviation already
            # gets large easily
            ENTROPY_LOSS_COEF = 0.005 #0.0001 #0.005 #0.01 #0.01

        # see: https://arxiv.org/abs/1711.02257
        # gradient normalization normalizes the gradient over all gradients together
        MAX_GRADIENT_NORMALIZATION = 0.5

        ### LEARNING PARAMETERS
        TOTAL_EPISODES_TO_TRAIN_BASE = 50

        # prediction is deterministic, actions are not sampled. This is explained in the thesis
        PREDICT_DETERMINISTIC = True

class paths:
    """
    This class stores the data paths for the results folders.
    This is all done automatically.
    """
    # ---------------LEAVE---------------
    # data paths
    DATA_PATH = "data"
    RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "preprocessed")

    # trained models and results path
    TRAINED_MODELS_PATH = "trained_models"
    RESULTS_PATH = "results"
    # names of sub-directories within results folder (based on the memories we save in the env.)
    SUBSUBDIR_NAMES = {"datadates": "datadates",
                       "cash_value": "cash_value",
                       "portfolio_value": "portfolio_value",
                       "rewards": "rewards",
                       "policy_actions": "policy_actions",
                       "policy_actions_trans": "policy_actions_trans",
                       "exercised_actions": "exercised_actions",
                       "asset_equity_weights": "asset_equity_weights",
                       "all_weights_cashAtEnd": "all_weights_cashAtEnd",
                       "transaction_cost": "transaction_cost",
                       "number_asset_holdings": "number_asset_holdings",
                       "sell_trades": "sell_trades",
                       "buy_trades": "buy_trades",
                       "state_memory": "state_memory",
                       "last_state": "last_state",
                       "backtest_bull": "backtest_bull",
                       "backtest_bear": "backtest_bear",
                       "training_performance": "training_performance",
                       }
    PREPROCESSED_DATA_FILE = os.path.join(PREPROCESSED_DATA_PATH, f"{settings.DATASET}.csv")