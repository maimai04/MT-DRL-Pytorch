import datetime
import os
import torch

"""
defining classes for grouping parameters.

classes:
--------
    settings
    paths
    env_params
"""

# TODO: update config loggings! e.g. which reward function used, 6.5. was log(newPFval/oldPFval) for mimicking risk aversion
class settings:
    """
    Defining general settings for the whole run and global variables.
    """
    # ---------------SET MANUALLY---------------
    # dataset used:
    DATASET = "US_stocks_WDB"
    #DATASET = "JP_stocks_WDB"$

    ### strategy mode to be run
    #STRATEGY_MODE = "ppo"
    STRATEGY_MODE = "ppoCustomBase"

    REWARD_MEASURE = "addPFVal" # additional portfolio value, = change in portfolio value as a eward
    #REWARD_MEASURE = "logU" # log utility of new / old value, in oder to "smooth out" larger rewards
    #REWARD_MEASURE = "SR7" # sharpe ratio, over 7 days # subtracting a volatility measure

    RETRAIN_DATA = False # = saving trained agent after each run and continue training only on the next train data chunk, using pre-trained agent (faster)
    #RETRAIN_DATA = True # = when training again on the whole training dataset for each episode

    ### Set dates
    # train
    STARTDATE_TRAIN = 20090101 #20141001 #20090102  # Note: this is also the "global startdate"
    ENDDATE_TRAIN = 20151231   #20151001
    # validation (only needed for get_data_params in preprocessing)
    #STARTDATE_VALIDATION = 20160101 #20151001
    #ENDDATE_VALIDATION = #20200707
    # trading starts on:     # 2016/01/01 is the date that real trading starts
    #STARTDATE_TRADE = 20160104
    #ENDDATE_TRADE = None
    # backtesting
    STARTDATE_BACKTESTING = 20000101
    ENDDATE_BACKTESTING = 20081231

    ### set rollover window; since we are doing rolling window / extended window cross validation for time series
    # 63 days = 3 months of each 21 trading days (common exchanges don't trade on weekends, need to change for crypto)
    ROLL_WINDOW = 63
    VALIDATION_WINDOW = 63
    TESTING_WINDOW = 63

    # ---------------LEAVE---------------
    ### define 10 randomly picked numbers to be used for seeding
    SEEDS_LIST = [0, 5]#, 23, 7774, 9090, 11112,  45252, 80923, 223445, 444110]
    SEED = None # placeholder, will be overwritten in run file)

    ### returns current timestamp, mainly used for naming directories/ printout / logging to .txt
    NOW = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    # this is going to be in the run folder name
    if RETRAIN_DATA:
        # if we retrain data, run modeis "long" (simply because it takes longer)
        RUN_MODE = "lng" # for "long"
    else:
        # if we do not retrain data, run mode is short (the run takes less long)
        RUN_MODE = "st" # for "short"

class data_settings:
    """
    Define variables and settings for data preprocessing.
    """
    # ---------------SET MANUALLY---------------
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
    RETURNS_FEATURES = ["log_return_daily"]
    TECH_INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30", "volume"]
    RISK_INDICATORS = ["ret_vola_7d"] #["returns_volatility"]

    # CHOOSE FEATURES MODE, BASED ON WHICH THE FEATURES LIST IS CREATED (SEE BELOW)
    FEATURES_MODE = "fm2"

    # ---------------LEAVE---------------
    if FEATURES_MODE == "fm1":
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES
    elif FEATURES_MODE == "fm2":
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES + TECH_INDICATORS
    elif FEATURES_MODE == "fm3":
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES + RISK_INDICATORS + TECH_INDICATORS
    elif FEATURES_MODE == "fm4":
        pass
    else:
        print("error (config): features list not found, cannot assign features mode.")

class env_params:
    # ---------------LEAVE---------------
    HMAX_NORMALIZE = 100  # This is the max. number of stocks one is allowed to buy of each stock
    INITIAL_CASH_BALANCE = 1000000
    TRANSACTION_FEE_PERCENT = 0.001  # todo: check reasonability: https://www.google.com/search?client=firefox-b-d&q=transaction+fee+for+stock+trading
    REWARD_SCALING = 1e-4  # This is 0.0001.

class agent_params:
    # ---------------SET MANUALLY---------------
    class ppo: # from stable baselines
        """
        POLICY          : policy network type, can be passed as str (if registered) or as instance,
                          e.g. if custom policy is used
                          "MlpPolicy": multiple linear perceptrons, standard neural network

        ENT_COEF        : entropy coefficient, how much we weight the entropy loss in the
        GAMMA           : discount factor
        LEARNING_RATE   : can also be variable, e.g. a function of the current progress remaining etc.
        N_STEPS         : number of steps the agent should take in the environment
                          The number of steps to run for each environment per update
                          (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
                          NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
                          See https://github.com/pytorch/pytorch/issues/29372
        BATCH_SIZE      : Optional[int] = 64, minibatch size
        N_EPOCHS        : number of epochs when optimizing the surrogate loss
        GAE_LAMBDa      : factor for trade-off of bias vs. variance for generalized advantage estimator
        CLIP_RANGE_VF   : clipping parameter for the value function
                          can be a function of current progress remaining (from 1 to 0)
                          NOTE: this parameter is specific to the OpenAI implementation
                          If None (default), no clipping will be one on the value function
                          IMPORTANT: this clipping depends on the reward scaling.
        CLIP_RANGE      : clipping parameter
                          can be a function of current progress remaining (from 1 to 0)
        VF_COEF         : value function coefficient for the loss calculation
        MAX_GRAD_NORM   : The maximum value for the gradient clipping
        USE_SDE         : Whether to use generalized State Dependent Exploration (gSDE)
                          instead of action noise exploration (default: False) # todo: ?
        SDE_SAMPLE_FREQ : Sample a new noise matrix every n steps when using gSDE
                          Default: -1 (only sample at the beginning of the rollout)
        TARGET_KL       : Limit the KL divergence between updates,
                          because the clipping is not enough to prevent large update
                          see issue #213, (cf https://github.com/hill-a/stable-baselines/issues/213)
                          By default, there is no limit on the kl div.
        TENSORBOARD_LOG : the log location for tensorboard (if None, no logging)
        CREATE_EVAL_ENV : Whether to create a second environment that will be used for evaluating
                          the agent periodically. (Only available when passing string for the environment)
        POLICY_KWARGS   : additional arguments to be passed to the policy on creation
        VERBOSE         : the verbosity level of logging: 0 no output, 1 info, 2 debug
        DEVICE          : DEVICE – DEVICE (cpu, cuda, ..., auto) on which the code should be run.
                          Setting it to auto, the code will be run on the GPU if possible.
        _INIT_SETUP_MODEL:

        Note: the parameters from the paper were default parameters from stable_baselines3, which were
        (I guess # todo: check)
        tune by the people from stable-baselines on stable-baselines zoo environment(s) # todo: find out which one
        Therefore, the parameters need to be tuned for the current evironment as well # todo!
        """
        ##### FEATURE EXTRACTOR & POLICY & VALUE NETWORKS
        # multiple linear perceptrons
        POLICY = 'MlpPolicy' # CustomMlp # CustomLSTM

        ##### HYPERPARAMETERS THAT NEED TO BEE TUNED
        # discount factor of returns: important because if high, future returns are discounted less and become more important
        # if discount rate is low, the opposite
        GAMMA = 0.99
        # clipping range for the policy loss (actor);
        # important because it ensures the new policy cannot be very different from the old one
        # the smaller te clip range, the more conservative we are with our policy changes
        CLIP_RANGE = 0.2
        # coefficient of value function (critic)
        # this is the weight the loss of the value network has in the whole network architecture
        # this is only relevant if we have a common network (part) for both critic (value function) and actor (policy function),
        # else the policy loss reduces to the policy loss only and the value loss is separate as well
        VF_COEF = 0.5
        # entropy coefficient for the loss calculation; how much we weight the entropy in the loss
        # entropy is added to the combined loss to ensure some exploration, that we don't get too stuck in a local optimum
        # if the entropy coefficient is too high, the agent might unlearn the things he learned to fast
        # see also: https://www.reddit.com/r/reinforcementlearning/comments/i3i5qa/ppo_to_high_entropy_coefficient_makes_agent/
        ENT_COEF = 0.005
        # generalized advantage estimate smoothing factor, the larger, the smoother our estimates (less variance)
        GAE_LAMBDA = 0.95

        ### NOT TUNED
        # BATCH_SIZE: Optional[int] = 64, # minibatch size for updating the network
        # Number of epoch when optimizing the network loss(es)
        N_EPOCHS = 10
        # maximal radient normalization
        MAX_GRAD_NORM = 0.5
        # number of timesteps that are collected into the buffer every time experience is collected
        # (there is not much to tune here because we have a limited data set anyways)
        N_STEPS = 2048
        # optimizer (neural network, by default: Adam) learning rate
        LEARNING_RATE = 3e-4 # not tuned because Adam already has some sort of adaptive learning rate and therefore is somewhat robust

        ### DEFAULT PARAMETERS / extended parameters, set to False / not tuned for this thesis
        CLIP_RANGE_VF = None # clipping range for the value function: as much as the policy loss can be clipped, the value function
        # can also be clipped; this is an extension not in the paper though
        # sde = state independent exploration, Fals eby default
        USE_SDE = False
        SDE_SAMPLE_FREQ = -1
        TARGET_KL = None
        TENSORBOARD_LOG = None #"TB_LOG"
        CREATE_EVAL_ENV = False
        POLICY_KWARGS = None
        VERBOSE = 0
        DEVICE = "auto"
        INIT_SETUP_MODEL = True

        ### HYPERPARAMETERS FOR PPO TRAINING
        # this is the number of total steps to be taken during training. The higher the number, the more often we are
        # going to sample new batches of data and train on them. Doing this too often might lead to "overtraining",
        # especially since we have a "fixed" stock market dataset. Doing too few steps might lead to te agent to
        # not learn enough. Since we are working with a rolling / expanding training window, it would make sense to make
        # the training time steps adaptive.
        TRAINING_TIMESTEPS = 10000 #100000

    class ppoCustomBase:
        """
        This class implements my own custom implementation of the PPO algorithm
        """
        ### SETUP PARAMETERS
        # net architecture mode
        NET_VERSION = "base1"
        #NET_VERSION = "base2"
        #NET_VERSION = "LSTM1"
        #NET_VERSION = "LSTM2"


        ### HYPERPARAMETERS
        BATCH_SIZE = 64
        NUM_EPOCHS = 10
        OPTIMIZER = torch.optim.Adam
        OPTIMIZER_LEARNING_RATE = 0.00025
        GAMMA = 0.99
        GAE_LAMBDA = 0.95
        CLIP_EPSILON = 0.1#0.5#0.2
        MAX_KL_VALUE = None
        CRITIC_LOSS_COEF = 0.5
        ENTROPY_LOSS_COEF = 0.01
        MAX_GRADIENT_NORMALIZATION = 0.5

        ### LEARNING PARAMETERS
        TOTAL_TIMESTEPS_TO_COLLECT = 5000 # normally set at length of train / vaidation / test data = > length of one episode
        TOTAL_TIMESTEPS_TO_TRAIN = 10000 #100000 # if > len(data), we will learn on the same data multiple times (but every time with different actions)

class paths:
    # ---------------LEAVE---------------
    # data paths
    DATA_PATH = "data"
    RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "preprocessed")  # todo: was PREPROCESSED_RAW_DATA_PATH

    # trained models and results path
    TRAINED_MODELS_PATH = "trained_models"
    RESULTS_PATH = "results"
    # names of sub-directories within results folder (based on the memories we save in the env.)
    SUBSUBDIR_NAMES = {"datadates": "datadates",
                       "cash_value": "cash_value",
                       "portfolio_value": "portfolio_value",
                       "rewards": "rewards",
                       "policy_actions": "policy_actions",
                       "exercised_actions": "exercised_actions",
                       "asset_equity_weights": "asset_equity_weights",
                       "all_weights_cashAtEnd": "all_weights_cashAtEnd",
                       "transaction_cost": "transaction_cost",
                       "number_asset_holdings": "number_asset_holdings",
                       "sell_trades": "sell_trades",
                       "buy_trades": "buy_trades",
                       "state_memory": "state_memory",
                       "last_state": "last_state",
                       "backtest": "backtest",
                       "training_performance": "training_performance",
                       }
    # data files # todo: rm
    #RAW_DATA_FILE = os.path.join(RAW_DATA_PATH, "dow_30_2009_2020.csv")

    # ---------------LEAVE---------------
    PREPROCESSED_DATA_FILE = os.path.join(PREPROCESSED_DATA_PATH, f"{data_settings.COUNTRY}_stocks_"
                                                                  f"{data_settings.DATABASE}.csv")
