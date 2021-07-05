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

# TODO: update config loggings! e.g. which reward function used, 6.5. was log(newPFval/oldPFval) for mimicking risk aversion
class settings:
    """
    Defining general settings for the whole run and global variables.
    """
    # dataset used:
    #DATASET = "US_stocks_WDB"
    #DATASET = "JP_stocks_WDB"$
    DATASET = "dummydata"

    obs_list_private = []
    # ---------------SET MANUALLY---------------
    ### strategy mode to be run
    STRATEGY_MODE = "ppo"
    #STRATEGY_MODE = "ddpg"
    #STRATEGY_MODE = "a2c"
    # STRATEGY_MODE = "ens1" # ensemble base (weight 1 for the best agent, weight 0 for each other agent)
    # STRATEGY_MODE = "ens2a" # ensemble weighted according to profitability, weights between 0 and 1
    # STRATEGY_MODE = "ens2b" # ensemble weighted according to profitability, weights can be zero (short-selling an agent)
    # STRATEGY_MODE = "ens3a" # ensemble weighted according to sharpe ratio, weights between 0 and 1
    # STRATEGY_MODE = "ens3b" # ensemble weighted according to sharpe ratio, weights can be zero (short-selling an agent)
    # if STRATEGY_MODE.find("ens") != -1:
    # if substring is found in main string, do:...
    # STRATEGY_MODE = None # only used for debugging (no models are run then)

    REWARD_MEASURE = "addPFVal"
    #REWARD_MEASURE = "logU"
    #REWARD_MEASURE = "SR7" # sharpe ratio, over 7 days

    #RUN_MODE = "st" # = "short",saving trained agent after each run and continue training only on the next train data chunk, using pre-trained agent (faster)
    RUN_MODE = "ext" # = "extended", when training again on the whole training dataset for each episode
    # NOTE: the modes above should give the same results actually, if the correct seed

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

    ### set windows
    # REBALANCE_WINDOW is the number of months to retrain the model
    # VALIDATION_WINDOW is the number of months to validate the model and select the DRL agent for trading
    # 63 days = 3 months of each 21 trading days (common exchanges don't trade on weekends, need to change for crypto)
    ROLL_WINDOW = 63
    # this is basically a "step" period; we take a step of 63 days for which we extend the training window and move the validation and trade window
    # todo: renamed to ROLL_WINDOW, was REBALANCE_WINDOW
    VALIDATION_WINDOW = 63
    TRAINING_WINDOW = 63
    TRADING_WINDOW = 63

    # ---------------LEAVE---------------
    if STRATEGY_MODE == "ppo" or STRATEGY_MODE == "ddpg" or STRATEGY_MODE == "a2c":
        AGENTS_LIST = [STRATEGY_MODE]

    ### set random seeds;
    SEEDS_LIST = [0, 5, 23, 7774, 9090, 11112,  45252, 80923, 223445, 444110]

    ### returns current timestamp, mainly used for naming directories/ printout / logging to .txt
    NOW = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


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
    TECH_INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30", "volume"]  # was FEATURES_LIST
    # TECH_INDICATORS = ["macd", "rsi", "cci", "adx"] # was FEATURES_LIST
    RISK_INDICATORS = ["ret_vola_7d"] #["returns_volatility"]
    ESG_INDICATORS = ["ESG"]
    LSTM_INDICATORS = []
    MARKET_INDICATORS = []

    # CHOOSE FEATURES MODE, BASED ON WHICH THE FEATURES LIST IS CREATED (SEE BELOW)
    #FEATURES_MODE = "fm1"
    FEATURES_MODE = "try"

    # ---------------LEAVE---------------
    if FEATURES_MODE == "fm1":
        FEATURES_LIST = PRICE_FEATURES
    elif FEATURES_MODE == "fm2":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS
    elif FEATURES_MODE == "fm3":
        FEATURES_LIST = PRICE_FEATURES + RISK_INDICATORS
    elif FEATURES_MODE == "fm4":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS + RISK_INDICATORS
    elif FEATURES_MODE == "fm5":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS + RISK_INDICATORS + RETURNS_FEATURES
    elif FEATURES_MODE == "fm6":
        FEATURES_LIST = RETURNS_FEATURES
    elif FEATURES_MODE == "fm7":
        FEATURES_LIST = RETURNS_FEATURES + TECH_INDICATORS
    elif FEATURES_MODE == "fm8":
        FEATURES_LIST = RETURNS_FEATURES + RISK_INDICATORS
    elif FEATURES_MODE == "fm9":
        FEATURES_LIST = RETURNS_FEATURES + TECH_INDICATORS + RISK_INDICATORS
    elif FEATURES_MODE == "fm10":
        FEATURES_LIST = RETURNS_FEATURES + TECH_INDICATORS + RISK_INDICATORS
    elif FEATURES_MODE == "try": # just for debugging
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES + TECH_INDICATORS + RISK_INDICATORS
    else:
        print("error (config): features list not found, cannot assign features mode.")

class env_params:
    # ---------------LEAVE---------------
    HMAX_NORMALIZE = 100  # todo: up and check perf
    INITIAL_CASH_BALANCE = 1000000
    TRANSACTION_FEE_PERCENT = 0.001  # todo: check reasonability: https://www.google.com/search?client=firefox-b-d&q=transaction+fee+for+stock+trading
    REWARD_SCALING = 1e-4  # This is 0.0001.
    # shorting possiblity? # todo

class agent_params:
    # ---------------SET MANUALLY---------------

    # ---------------LEAVE---------------
    class _a2c:
        POLICY = 'MlpPolicy'
        TRAINING_TIMESTEPS = 25000
        # default values (given by stable-baselines)
        LEARNING_RATE = 7e-4
        N_STEPS = 5
        GAMMA = 0.99
        GAE_LAMBDA = 1.0
        ENT_COEF = 0.0
        VF_COEF = 0.5
        MAX_GRAD_NORM = 0.5
        RMS_PROP_EPS = 1E-5
        USE_RMS_PROP = True
        USE_SDE = False
        SDE_SAMPLE_FREQ = -1
        NORMALIZE_ADVANTAGE = False
        TENSORBOARD_LOG = None# Optional
        CREATE_EVAL_ENV = False
        POLICY_KWARGS = None # optional
        VERBOSE = 0
        DEVICE = "auto"
        INIT_SETUP_MODEL = True

    class _ddpg:
        """

        """
        POLICY = 'MlpPolicy'
        # param_noise=param_noise, # todo: was in paper version, was None
        ACTION_NOISE = "OUAN"  # OUAN for OhrsteinUhlenbeckActionNoise  # default: None
        # default values (given by stable-baselines)
        LEARNING_RATE = 1e-3
        BUFFER_SIZE = int(1e6)
        LEARNING_STARTS = 100
        BATCH_SIZE = 100
        TAU = 0.005
        GAMMA = 0.99
        TRAIN_FREQ = (1, "episode")
        GRADIENT_STEPS = -1
        OPTIMIZE_MEMORY_USAGE = True # Default: was False
        TENSORBOARD_LOG = None
        CREATE_EVAL_ENV = False
        POLICY_KWARGS = None
        VERBOSE = 0
        DEVICE = "auto"
        INIT_SETUP_MODEL = True
        TRAINING_TIMESTEPS = 10000 # otherwise takes too long? if 100'000

    class _ppo:
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
        POLICY = 'MlpPolicy'

        ##### HYPERPARAMETERS THAT NEED TO BEE TUNED
        # entropy coefficient for the loss calculation; how much we weight the entropy in the loss
        ENT_COEF = 0.005

        GAMMA = 0.99
        LEARNING_RATE = 3e-4
        N_STEPS = 2048 # number of steps to run for one update of the (actor/critic) network parameters, default = 2048 # todo: ?
                       # rollout buffer size = n_steps here (since only 1 environment used)
                        # = n_rollout_steps
        # BATCH_SIZE: Optional[int] = 64, # minibatch size
        N_EPOCHS = 10 # Number of epoch when optimizing the network loss(es)
        GAE_LAMBDA = 0.95
        CLIP_RANGE_VF = None
        CLIP_RANGE = 0.2
        VF_COEF = 0.5
        MAX_GRAD_NORM = 0.5
        USE_SDE = False
        SDE_SAMPLE_FREQ = -1
        TARGET_KL = None
        TENSORBOARD_LOG = "TB_LOG"
        CREATE_EVAL_ENV = False
        POLICY_KWARGS = None
        VERBOSE = 0
        DEVICE = "auto"
        INIT_SETUP_MODEL = True

        ### HYPERPARAMETERS FOR PPO TRAINING
        TRAINING_TIMESTEPS = 10000#100000 # todo: ?

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
                       "transaction_cost": "transaction_cost",
                       "number_asset_holdings": "number_asset_holdings",
                       "sell_trades": "sell_trades",
                       "buy_trades": "buy_trades",
                       "crisis_measures": "crisis_measures",
                       "crisis_thresholds": "crisis_thresholds",
                       "crisis_selloff_cease_trading": "crisis_selloff_cease_trading",
                       "state_memory": "state_memory",
                       "last_state": "last_state",
                       "backtest": "backtest",
                       }
    # data files
    RAW_DATA_FILE = os.path.join(RAW_DATA_PATH, "dow_30_2009_2020.csv")  # todo: was TRAINING_DATA_FILE
    # ---------------LEAVE---------------
    if data_settings.DATABASE == "WhartonDB" and data_settings.FEATURES_MODE == "fm1":
        PREPROCESSED_DATA_FILE = os.path.join(PREPROCESSED_DATA_PATH, DATASET +".csv")
    else:
        PREPROCESSED_DATA_FILE = os.path.join(PREPROCESSED_DATA_PATH, f"{data_settings.COUNTRY}_stocks_"
                                                                      f"{data_settings.DATABASE}_"
                                                                      f"{data_settings.FEATURES_MODE}.csv")

class crisis_settings:
    """
    Choose if you want to use a crisis measure (such as turbulence and other) to act as a stop loss in times of
    turbulence.
    Choose parameters and settings for your crisis measure of choice.
    """

    # ---------------SET MANUALLY---------------
    CRISIS_MEASURE = None  # default
    # CRISIS_MEASURE = "turbulence"
    # CRISIS_MEASURE = "volatility"

    # ---------------LEAVE---------------
    if CRISIS_MEASURE == "turbulence":
        CNAME = "turb"
        # CRISIS_THRESHOLD = 140  # turbulence threshold
        # CRISIS_DATA = None  # path to pre-calculated data
        CUTOFF_XPERCENTILE = .90
        print("(config) crisis condition measure: {}".format(CRISIS_MEASURE))
    elif CRISIS_MEASURE == "volatility":
        CNAME = "vola"
        CUTOFF_XPERCENTILE = ""
        # CRISIS_THRESHOLD = None  # turbulence threshold
        # CRISIS_DATA = None
        print("(config) crisis condition measure: {}".format(CRISIS_MEASURE))
    elif CRISIS_MEASURE is None:
        CNAME = ""
        CUTOFF_XPERCENTILE = ""
        print("(config) no crisis measure selected.")
    else:
        print("(config) ValueError: crisis measure selected unkNOWn and not None.")