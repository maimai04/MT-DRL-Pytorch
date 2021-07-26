import sys
from pipeline.run_pipeline import *
from config.config import *
from config.dataprep_config import *

######################################################
##   DEFINING FUNCTIONS USED IN the run.py file     ##
######################################################

def load_dataset(*,
                 file_path: str,
                 col_subset: list,
                 date_subset: str = "datadate",
                 date_subset_startdate: int = 19950101,
                 asset_name_column: str="tic",
                 ) -> pd.DataFrame:
    """
    Load the .csv dataset from the provided file path.

    INPUT: path to df as csv, ordered by ticker, then date
    OUTPUT: df as pd.DataFrame(), ordered by ticker, then date

    @param file_path: the path to the preprocessed data
    @param col_subset: the columns we want to use as feature in this run
    @param date_subset: the name of the date column
    @param date_subset_startdate: the date from which on we want to import the dataset
    @param asset_name_column: name of the asset name column, where tickers are stored
    @return:

    # Note: the asterisk (*) enforces that all the following variables have to be specified as keyword argument, when being called
    # see also: https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/
    """
    # import df
    df = pd.read_csv(file_path, index_col=0)
    # if we have specified the name of the date column and  provided a startdate,
    # we get the df subset based on these params
    if date_subset and date_subset_startdate is not None:
        df = df[df[date_subset] >= date_subset_startdate]
    # if we have specified a list of column names (features), we subset them
    if col_subset is not None:
        subcols = [date_subset, asset_name_column] + col_subset
        df = df[subcols]
    return df

def create_dirs(mode: str = "run_dir", # "seed_dir"
                results_dir: str = "",
                trained_dir: str = "",
                results_path: str = "results",
                trained_models_path: str = "trained_models",
                now: str = "",
                strategy_mode: str = "ppoCustomBase",
                features_mode: str = "",
                reward_measure: str = "",
                net_arch: str = "",
                env_step_version: str = "",
                run_mode: str = "",
                seed: int=None,
                subdir_names: dict = {},
                predict_deterministic: bool=False
                ) -> list:
    """

    @param mode: "run_dir" if we are creating directories for the whole run,
                 "seed_dir" if we are creating directories for the run for a specific seed (which is a sub-run of the whole run)
    @param results_dir: path to where the results directory should be created
    @param trained_dir: path to where the directory for storing trained models should be created
    @param results_path: name of the overall results folder where all results for all runs should be stored
    @param trained_models_path: name of the overall trained models folder folder where all trained models for all runs should be stored

    The below variables are only used to create the foldername for the run, they are set in the config.py file
    @param now: date and time of the current run, automatically created in config.py
    @param strategy_mode: ---
    @param features_mode: used fo
    @param reward_measure:
    @param net_arch:
    @param env_step_version:
    @param run_mode:
    @param seed:
    @param subdir_names:
    @param predict_deterministic:
    """
    # for the whole run
    if mode == "run_dir":
        ### RESULTY DIRECTORY
        # creating results directory for the current run folders (one run/folder for each seed in this directory)
        results_dir = os.path.join(results_path,
                                   f"{now}"
                                   f"_{strategy_mode}"
                                   f"_{reward_measure}"
                                   f"_{net_arch}"
                                   f"_{env_step_version}"
                                   f"_{features_mode}"
                                   f"_{run_mode}"
                                   f"_det{predict_deterministic}")
        os.makedirs(results_dir) # os.makedirs() method creates all unavailable/missing directories under the specified path

        ### TRAINED MODEL DIRECTORY (saving the trained DRL models)
        trained_dir = os.path.join(trained_models_path, f"{now}"
                                                        f"_{strategy_mode}"
                                                        f"_{reward_measure}"
                                                        f"_{net_arch}"
                                                        f"_{env_step_version}"
                                                        f"_{features_mode}"
                                                        f"_{run_mode}"
                                                        f"_det{predict_deterministic}")
        os.makedirs(trained_dir)
        return [results_dir, trained_dir]

    # for one seed (seed-run) within a whole run
    if mode == "seed_dir":
        ### RESULTS SUBDIRECTORY
        # creating results sub-directory, one for each seed (for which the algorithm is run) within one run
        results_subdir = os.path.join(results_dir, f"randomSeed{seed}")
        os.makedirs(results_subdir)
        ### TRAINED MODEL SUBDIRECTORY
        trained_subdir = os.path.join(trained_dir, f"randomSeed{seed}")
        os.makedirs(trained_subdir)

        # creating sub-sub-directories for the actual results folders (e.g. portfolio_value etc.)
        # where the resulting .csv files are saved during the run
        # the names of the sub-sub-directories are defined in the config file under paths.SUBSUBDIR_NAMES
        for dirname in subdir_names.keys():
            subdir_path = os.path.join(results_subdir, dirname)
            os.makedirs(subdir_path)
        del subdir_path
        return [results_subdir, trained_subdir]

def config_logging_to_txt(results_subdir,
                          trained_subdir,
                          logsave_path,
                          now,
                          seeds_list,
                          strategy_mode,
                          reward_measure,
                          retrain_data,
                          run_mode,
                          global_startdate_train,
                          global_enddate_train,
                          roll_window,
                          startdate_backtesting_bull,
                          enddate_backtesting_bull,
                          startdate_backtesting_bear,
                          enddate_backtesting_bear,
                          env_step_version,
                          hmax_normalize,
                          initial_cash_balance,
                          transaction_fee,
                          reward_scaling,
                          rebalance_penalty,
                          country,
                          features_list,
                          single_features_list,
                          lstm_features_list,
                          features_mode,
                          data_path,
                          raw_data_path,
                          preprocessed_data_path,
                          trained_models_path ,
                          results_path,
                          subdir_names,
                          preprocessed_data_file,
                          net_arch,
                          batch_size,
                          num_epochs,
                          optimizer,
                          optimizer_lr,
                          gamma,
                          gae_lam,
                          clip_epsilon,
                          critic_loss_coef,
                          entropy_loss_coef,
                          max_gradient_norm,
                          total_episodes_to_train_base,
                          predict_deterministic,
                          ) -> None:
    """
    Writes all configurations and related parameters into the config_log.txt file.
    This is done for each run, so that every run can be uniquely identified based
    on its configurations.

    The parameters are all the parameters from the config.py file, hence they are not explained again.
    """
    txtfile_path = os.path.join(logsave_path, "configurations.txt")

    with open(txtfile_path, "w") as text_file:
        text_file.write("------------------------------------\n"
                        "CONFIGURATION SETTINGS AND VARIABLES\n"
                        "------------------------------------\n"
                        "------------------------------------\n"
                        "SETTINGS\n"
                        "------------------------------------\n"
                        f"NOW                  : {now}\n"
                        f"SEEDS LIST           : {seeds_list}\n"
                        f"STRATEGY_MODE        : {strategy_mode}\n"
                        f"REWARD_MEASURE       : {reward_measure}\n"
                        f"RETRAIN_DATA         : {retrain_data}\n"
                        f"RUN_MODE             : {run_mode}\n"
                        f"STARTDATE_TRAIN      : {global_startdate_train}\n"
                        f"ENDDATE_TRAIN        : {global_enddate_train}\n"
                        f"ROLL_WINDOW          : {roll_window}\n"
                        f"STARTDATE_BT_BULL    : {startdate_backtesting_bull}\n"
                        f"ENDDATE_BT_BULL      : {enddate_backtesting_bull}\n"
                        f"STARTDATE_BT_BEAR    : {startdate_backtesting_bear}\n"
                        f"ENDDATE_BT_BEAR      : {enddate_backtesting_bear}\n"
                        "------------------------------------\n"
                        f"ENVIRONMENT VARIABLES\n"
                        "------------------------------------\n"
                        f"STEP_VERSION             : {env_step_version}\n"
                        f"HMAX_NORMALIZE           : {hmax_normalize}\n"
                        f"INITIAL_CASH_BALANCE     : {initial_cash_balance}\n"
                        f"TRANSACTION_FEE_PERCENT  : {transaction_fee}\n"
                        f"REWARD_SCALING           : {reward_scaling}\n"
                        f"REBALANCE_PENALTY        : {rebalance_penalty}\n"
                        "------------------------------------\n"
                        f"DATA PREPARATION SETTINGS\n"
                        "------------------------------------\n"
                        f"DATA / COUNTRY           : {country}\n"
                        f"FEATURES_LIST            : {features_list}\n"
                        f"SINGLE_FEATURES_LIST     : {single_features_list}\n"
                        f"LSTM_FEATURES_LIST       : {lstm_features_list}\n"
                        f"FEATURES_MODE            : {features_mode}\n"
                        "------------------------------------\n"
                        f"PATHS AND DIRECTORIES\n"
                        "------------------------------------\n"
                        f"DATA_PATH                : {data_path}\n"
                        f"RAW_DATA_PATH            : {raw_data_path}\n"
                        f"PREPROCESSED_DATA_PATH   : {preprocessed_data_path}\n"
                        f"TRAINED_MODELS_PATH      : {trained_models_path}\n"
                        f"RESULTS_PATH             : {results_path}\n"
                        f"SUBDIR_NAMES             : {subdir_names}\n"
                        f"PREPROCESSED_DATA_FILE   : {preprocessed_data_file}\n"
                        f"RESULTS_DIR              : {results_subdir}\n"
                        f"TRAINED_MODEL_DIR        : {trained_subdir}\n"
                        "------------------------------------\n"
                        f"PPO AGENT PARAMETERS\n"
                        "------------------------------------\n"
                        f"NET_VERSION                   : {net_arch}\n"
                        f"BATCH_SIZE                    : {batch_size}\n"
                        f"NUM_EPOCHS                    : {num_epochs}\n"
                        f"OPTIMIZER                     : {optimizer}\n"
                        f"OPTIMIZER_LEARNING_RATE       : {optimizer_lr}\n"
                        f"GAMMA                         : {gamma}\n"
                        f"GAE_LAMBDA                    : {gae_lam}\n"
                        f"CLIP_EPSILON                  : {clip_epsilon}\n"
                        f"CRITIC_LOSS_COEF              : {critic_loss_coef}\n"
                        f"ENTROPY_LOSS_COEF             : {entropy_loss_coef}\n"
                        f"MAX_GRADIENT_NORMALIZATION    : {max_gradient_norm}\n"
                        f"TOTAL_TIMESTEPS_TO_COLLECT (Cont)    : {total_episodes_to_train_base}\n"
                        f"PREDICT_DETERMINISTIC         : {predict_deterministic}\n"
                        )
    return None

def custom_logger(seed: int,
                  logging_path: str,
                  level: logging = logging.NOTSET):
    """
    Method to return a custom logger with the given name and level.
    Used for logging to log files during the run, for each seed, all saved in _LOGGINGS.
    """
    logger_name = f"run_log_seed_{seed}"
    filename = os.path.join(logging_path, logger_name+".txt")
    logger = logging.getLogger(filename)
    logger.setLevel(level)
    format_string = '%(asctime)s %(levelname)s %(message)s'
    log_format = logging.Formatter(format_string)
    filemode = 'a'
    file_handler = logging.FileHandler(filename, mode=filemode)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def get_data_params(final_df: pd.DataFrame, # imported at the beginning of each run
                    feature_cols: list = [], # defined in config
                    single_feature_cols: list = [],# defined in config
                    lstm_cols: list = [], # defined in config
                    asset_name_column="tic", # defined in config but also default
                    ):
    """
    Get some parameters we need, based on the final pre-processed dataset:
        number of individual assets (n_individual_assets)
        number of features used (n_features)
        number of single features (n_single_features), here only VIX
        number of LSTM features (n_features_lstm)
        number of single LSTM fetaures (n_single_features_lstm), here only VIX
    """
    df = final_df.copy()
    n_individual_assets = len(df[asset_name_column].unique())
    n_features = len(feature_cols)
    n_single_features = len(single_feature_cols)

    # for lstm features: # get intersection between whole features list and lstm list to find lstm asset features
    set_features_cols = set(feature_cols)
    lstm_features_cols = list(set_features_cols.intersection(lstm_cols))
    n_features_lstm = len(lstm_features_cols)
    # for lstm single features # get intersection between single feature list and lstm list to find single lstm features
    set_single_feature_cols = set(single_feature_cols)
    lstm_single_features_cols = list(set_single_feature_cols.intersection(lstm_cols))
    n_single_features_lstm = len(lstm_single_features_cols)

    return n_individual_assets, n_features, n_single_features, n_features_lstm, n_single_features_lstm

