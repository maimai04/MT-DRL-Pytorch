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
    If a col_subset is specified, only the specified columns subset of the loaded dataset is returned.
    (This can be used if there are many columns and we only want to use 5 of them e.g.)

    @param col_subset:
    @param date_subset_startdate:
    @param date_subset:
    @type file_path: object
    @param file_path as specified in config.py
    @return: (df) pandas dataframe

    # Note: the asterisk (*) enforces that all the following variables have to be specified as keyword argument, when being called
    # see also: https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/

    INPUT: df as csv, ordered by ticker, then date
    OUTPUT: df as pd.DataFrame(), ordered by ticker, then date
    """
    df = pd.read_csv(file_path, index_col=0)

    if date_subset and date_subset_startdate is not None:
        df = df[df[date_subset] >= date_subset_startdate]

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
                net_version: str = "",
                env_step_version: str = "",
                run_mode: str = "",
                seed: int=None,
                subdir_names: dict = {},
                ) -> list:
    """
    Functin to create directories at the beginning of each run, based on paths specified in the config.py file,
    used in the run.py file.

    @param mode: run_dir - creates a directory for results and trained models of whole run (all seeds).
        seed_dir - created a directory (within the directory for results resp. trained models) for each seed.
    @param results_dir: name of results directory.
    @param trained_dir: name of trained models directory.
    @return:
    """
    if mode == "run_dir":
        ### RESULTY DIRECTORY
        # creating results directory for the current run folders (one run/folder for each seed in this directory)
        results_dir = os.path.join(results_path,
                                   f"{now}"
                                   f"_{strategy_mode}"
                                   f"_{reward_measure}"
                                   f"_{net_version}"
                                   f"_{env_step_version}"
                                   f"_{features_mode}"
                                   f"_{run_mode}")
        os.makedirs(results_dir) # os.makedirs() method creates all unavailable/missing directories under the specified path

        ### TRAINED MODEL DIRECTORY (saving the trained DRL models)
        trained_dir = os.path.join(trained_models_path, f"{now}"
                                                        f"_{strategy_mode}"
                                                        f"_{reward_measure}"
                                                        f"_{net_version}"
                                                        f"_{env_step_version}"
                                                        f"_{features_mode}"
                                                        f"_{run_mode}")
        os.makedirs(trained_dir)
        return [results_dir, trained_dir]

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
                          now_hptuning,
                          only_hptuning,
                          gamma_list,
                          gae_lam_list,
                          clip_list,
                          critic_loss_coef_hpt,
                          entropy_loss_coef_hpt,
                          net_version,
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
                          total_episodes_to_train_cont,
                          ) -> None:
    """
    Writes all configurations and related parameters into the config_log.txt file.

    @param results_subdir: the directory where the results for the current run are saved
    @param trained_subdir: the directory where the model of the current run is saved
    @param logsave_path: where the hereby created logging file is saved
    @return: None
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
                        #"------------------------------------\n"
                        #f"HYPERPARAMETER TUNING\n"
                        #"------------------------------------\n"
                        #f"now_hptuning              : {now_hptuning}\n"
                        #f"only_hptuning             : {only_hptuning}\n"
                        #f"--parameters to tune--\n"
                        #f"GAMMA_LIST                : {gamma_list}\n"
                        #f"GAE_LAMBDA_LIST           : {gae_lam_list}\n"
                        #f"CLIP_EPSILON_LIST         : {clip_list}\n"
                        #f"CRITIC_LOSS_COEF_LIST     : {critic_loss_coef_hpt}\n"
                        #f"ENTROPY_LOSS_COEF_LIST    : {entropy_loss_coef_hpt}\n"
                        )

    if settings.STRATEGY_MODE == "ppoCustomBase":
        with open(txtfile_path, "a") as text_file:
            text_file.write(f"NET_VERSION                   : {net_version}\n"
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
                            f"TOTAL_TIMESTEPS_TO_COLLECT (Base)    : {total_episodes_to_train_base}\n"
                            f"TOTAL_TIMESTEPS_TO_COLLECT (Cont)    : {total_episodes_to_train_cont}\n"
                            )
    elif settings.STRATEGY_MODE == "ppo": # todo: rm
        with open(txtfile_path, "a") as text_file:
            text_file.write(f"POLICY             : {agent_params.ppo.POLICY}\n"
                            f"ENT_COEF           : {agent_params.ppo.ENT_COEF}\n"
                            f"GAMMA              : {agent_params.ppo.GAMMA}\n"
                            f"LEARNING_RATE      : {agent_params.ppo.LEARNING_RATE}\n"
                            f"N_STEPS (buffer)   : {agent_params.ppo.N_STEPS}\n"
                            f"BATCH_SIZE         : Optional[int] = 64\n"
                            f"N_EPOCHS           : {agent_params.ppo.N_EPOCHS}\n"
                            f"GAE_LAMBDA         : {agent_params.ppo.GAE_LAMBDA}\n"
                            f"CLIP_RANGE         : {agent_params.ppo.CLIP_RANGE}\n"
                            f"VF_COEF            : {agent_params.ppo.VF_COEF}\n"
                            f"default parameters (unchanged from default as given by stable-baselines):"
                            f"MAX_GRAD_NORM      : {agent_params.ppo.MAX_GRAD_NORM}\n"
                            f"CLIP_RANGE_VF      : {agent_params.ppo.CLIP_RANGE_VF}\n"
                            f"USE_SDE            : {agent_params.ppo.USE_SDE}\n"
                            f"SDE_SAMPLE_FREQ    : {agent_params.ppo.SDE_SAMPLE_FREQ}\n"
                            f"TARGET_KL          : {agent_params.ppo.TARGET_KL}\n"
                            f"TENSORBOARD_LOG    : {agent_params.ppo.TENSORBOARD_LOG}\n"
                            f"CREATE_EVAL_ENV    : {agent_params.ppo.CREATE_EVAL_ENV}\n"
                            f"POLICY_KWARGS      : {agent_params.ppo.POLICY_KWARGS}\n"
                            f"VERBOSE            : {agent_params.ppo.VERBOSE}\n"
                            f"DEVICE             : {agent_params.ppo.DEVICE}\n"
                            f"INIT_SETUP_MODEL   : {agent_params.ppo.INIT_SETUP_MODEL}\n"
                            f"TRAINING_TIMESTEPS : {agent_params.ppo.TRAINING_TIMESTEPS}\n")
    return None

def custom_logger(seed: int,
                  logging_path: str,
                  level: logging = logging.NOTSET):
    """
    Method to return a custom logger with the given name and level
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

def get_data_params(final_df: pd.DataFrame,
                    feature_cols: list = [],
                    single_feature_cols: list = [],
                    lstm_cols: list = [],
                    asset_name_column="tic",
                    ):
    """
    Get some parameters we need, based on the final pre-processed dataset:
        number of individual assets (n_individual_assets)
        number of features used (n_features)
        unique trade dates within the wished validation (or other) subset (unique_trade_dates)
    @param final_df:
    @param asset_name_column:
    @return:
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

