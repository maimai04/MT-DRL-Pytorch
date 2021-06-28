import os
import pandas as pd
from model.models_pipeline import *
from config.config import *
from config.dataprep_config import *

######################################################
##   DEFINING FUNCTIONS USED IN the run.py file     ##
######################################################

def load_dataset(*,
                 file_path: str,
                 col_subset: list = data_settings.FEATURES_LIST,
                 date_subset: str = "datadate",
                 date_subset_startdate: int = 19950101,
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
        subcols = [date_subset, data_settings.ASSET_NAME_COLUMN] + col_subset
        df = df[subcols]

    return df

def create_dirs(mode: str = "run_dir", # "seed_dir"
                results_dir: str = "",
                trained_dir: str = "",
                dummydata: bool = False,
                ) -> list:
    # todo: create support_functions and move there
    """
    Functin to create directories at the beginning of each run, based on paths specified in the config.py file,
    used in the run.py file.

    @param mode: run_dir - creates a directory for results and trained models of whole run (all seeds).
        seed_dir - created a directory (within the directory for results resp. trained models) for each seed.
    @param results_dir: name of results directory.
    @param trained_dir: name of trained models directory.
    @return:
    """
    if dummydata:
        results_path = os.path.join("dummyresults", "results")
        trained_models_path = os.path.join("dummyresults", "trained_models")
    else:
        results_path = paths.RESULTS_PATH
        trained_models_path = paths.TRAINED_MODELS_PATH

    if mode == "run_dir":
        ### RESULTY DIRECTORY
        # creating results directory for the current run folders (one run/folder for each seed in this directory)
        results_dir = os.path.join(results_path,
                                   f"{settings.NOW}_{settings.STRATEGY_MODE}_"
                                   f"{crisis_settings.CNAME}_{data_settings.FEATURES_MODE}"
                                   f"_{settings.RUN_MODE}")
        os.makedirs(results_dir) # os.makedirs() method creates all unavailable/missing directories under the specified path

        ### TRAINED MODEL DIRECTORY (saving the trained DRL models)
        trained_dir = os.path.join(trained_models_path, f"{settings.NOW}"
                                                              f"_{settings.STRATEGY_MODE}"
                                                              f"_{crisis_settings.CNAME}"
                                                              f"_rew_{settings.REWARD_MEASURE}"
                                                              f"_{data_settings.FEATURES_MODE}"
                                                              f"_{settings.RUN_MODE}")
        os.makedirs(trained_dir)
        return [results_dir, trained_dir]

    if mode == "seed_dir":
        ### RESULTY SUBDIRECTORY
        # creating results sub-directory, one for each seed (for which the algorithm is run) within one run
        results_subdir = os.path.join(results_dir, f"agentSeed{settings.SEED_AGENT}")
        os.makedirs(results_subdir)
        ### TRAINED MODEL SUBDIRECTORY
        trained_subdir = os.path.join(trained_dir, f"agentSeed{settings.SEED_AGENT}")
        os.makedirs(trained_subdir)

        # creating sub-sub-directories for the actual results folders (e.g. portfolio_value etc.)
        # where the resulting .csv files are saved during the run
        # the names of the sub-sub-directories are defined in the config file under paths.SUBSUBDIR_NAMES
        if crisis_settings.CRISIS_MEASURE is None and "crisis_measures" in paths.SUBSUBDIR_NAMES:
            [paths.SUBSUBDIR_NAMES.pop(key) for key in ["crisis_measures", "crisis_thresholds", "crisis_selloff_cease_trading"]]
        for dirname in paths.SUBSUBDIR_NAMES.keys():
            subdir_path = os.path.join(results_subdir, dirname)
            os.makedirs(subdir_path)
        del subdir_path
        return [results_subdir, trained_subdir]

def config_logging_to_txt(results_subdir,
                          trained_subdir,
                          logsave_path
                          ) -> None:
    # todo: create support_functions and move there
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
                        f"NOW                  : {settings.NOW}\n"
                        f"SEEDS LIST           : {settings.SEEDS_LIST}\n"
                        f"STRATEGY_MODE        : {settings.STRATEGY_MODE}\n"
                        f"AGENTS_LIST          : {settings.AGENTS_LIST}\n"
                        f"ROLLING_WINDOW       : {settings.ROLL_WINDOW}\n"
                        f"VALIDATION_WINDOW    : {settings.VALIDATION_WINDOW}\n"
                        f"STARTDATE_TRAIN      : {settings.STARTDATE_TRAIN}\n"
                        f"ENDDATE_TRAIN        : {settings.ENDDATE_TRAIN}\n"
                       # f"STARTDATE_VALIDATION : {settings.STARTDATE_VALIDATION}\n"
                       # f"ENDDATE_VALIDATION   : {settings.ENDDATE_VALIDATION}\n"
                        "------------------------------------\n"
                        f"ENVIRONMENT VARIABLES\n"
                        "------------------------------------\n"
                        f"HMAX_NORMALIZE           : {env_params.HMAX_NORMALIZE}\n"
                        f"INITIAL_CASH_BALANCE     : {env_params.INITIAL_CASH_BALANCE}\n"
                        f"TRANSACTION_FEE_PERCENT  : {env_params.TRANSACTION_FEE_PERCENT}\n"
                        f"REWARD_SCALING           : {env_params.REWARD_SCALING}\n"
                        "------------------------------------\n"
                        f"DATA PREPARATION SETTINGS\n"
                        "------------------------------------\n"
                        f"FEATURES_LIST    : {data_settings.FEATURES_LIST}\n"
                        f"FEATURES_MODE    : {data_settings.FEATURES_MODE}\n"
                        "------------------------------------\n"
                        f"CRISIS SETTINGS\n"
                        "------------------------------------\n"
                        f"CRISIS_MEASURE: {crisis_settings.CRISIS_MEASURE}\n"
                        "------------------------------------\n"
                        f"PATHS AND DIRECTORIES\n"
                        "------------------------------------\n"
                        f"DATA_PATH                : {paths.DATA_PATH}\n"
                        f"RAW_DATA_PATH            : {paths.RAW_DATA_PATH}\n"
                        f"PREPROCESSED_DATA_PATH   : {paths.PREPROCESSED_DATA_PATH}\n"
                        f"TRAINED_MODELS_PATH      : {paths.TRAINED_MODELS_PATH}\n"
                        f"RESULTS_PATH             : {paths.RESULTS_PATH}\n"
                        f"SUBDIR_NAMES             : {paths.SUBSUBDIR_NAMES}\n"
                        f"TESTING_DATA_FILE        : {paths.TESTING_DATA_FILE}\n" # todo: where is this file?
                        f"RAW_DATA_FILE            : {paths.RAW_DATA_FILE}\n"
                        f"PREPROCESSED_DATA_FILE   : {paths.PREPROCESSED_DATA_FILE}\n"
                        f"RESULTS_DIR              : {results_subdir}\n"
                        f"TRAINED_MODEL_DIR        : {trained_subdir}\n")  # trained_model_dir
    for agent_ in settings.AGENTS_LIST:
        with open(txtfile_path, "a") as text_file:
            text_file.write("------------------------------------\n"
                            f"AGENT PARAMETERS - - - {agent_.upper()}\n"
                            "------------------------------------\n")
        if agent_ == "ppo":
            with open(txtfile_path, "a") as text_file:
                text_file.write(f"policy             : {agent_params._ppo.POLICY}\n"
                                f"ent_coef           : {agent_params._ppo.ENT_COEF}\n"
                                f"default parameters (unchanged from default as given by stable-baselines):"
                                f"gamma              : {agent_params._ppo.GAMMA}\n"
                                f"learning_rate      : {agent_params._ppo.LEARNING_RATE}\n"
                                f"n_steps            : {agent_params._ppo.N_STEPS}\n"
                                f"batch_size         : Optional[int] = 64\n"
                                f"n_epochs           : {agent_params._ppo.N_EPOCHS}\n"
                                f"gae_lambda         : {agent_params._ppo.GAE_LAMBDA}\n"
                                f"clip_range_vf      : {agent_params._ppo.CLIP_RANGE_VF}\n"
                                f"clip_range         : {agent_params._ppo.CLIP_RANGE}\n"
                                f"vf_coef            : {agent_params._ppo.VF_COEF}\n"
                                f"max_grad_norm      : {agent_params._ppo.MAX_GRAD_NORM}\n"
                                f"use_sde            : {agent_params._ppo.USE_SDE}\n"
                                f"sde_sample_freq    : {agent_params._ppo.SDE_SAMPLE_FREQ}\n"
                                f"target_kl          : {agent_params._ppo.TARGET_KL}\n"
                                f"tensorboard_log    : {agent_params._ppo.TENSORBOARD_LOG}\n"
                                f"create_eval_env    : {agent_params._ppo.CREATE_EVAL_ENV}\n"
                                f"policy_kwargs      : {agent_params._ppo.POLICY_KWARGS}\n"
                                f"verbose            : {agent_params._ppo.VERBOSE}\n"
                                f"device             : {agent_params._ppo.DEVICE}\n"
                                f"init_setup_model   : {agent_params._ppo.INIT_SETUP_MODEL}\n"
                                f"TRAINING_TIMESTEPS : {agent_params._ppo.TRAINING_TIMESTEPS}\n")
        elif agent_ == "ddpg":
            with open(txtfile_path, "a") as text_file:
                text_file.write(f"policy             : {agent_params._ddpg.POLICY}\n"
                                f"action_noise       : {agent_params._ddpg.ACTION_NOISE}\n"
                                f"default parameters (unchanged from default as given by stable-baselines):"
                                f"gamma              : {agent_params._ddpg.GAMMA}\n"
                                f"learning_rate      : {agent_params._ddpg.LEARNING_RATE}\n"
                                f"buffer_size        : {agent_params._ddpg.BUFFER_SIZE}\n"
                                f"learning_starts    : {agent_params._ddpg.LEARNING_STARTS}\n"
                                f"batch_size         : {agent_params._ddpg.BATCH_SIZE}\n"
                                f"tau                : {agent_params._ddpg.TAU}\n"
                                f"gradient_steps     : {agent_params._ddpg.GRADIENT_STEPS}\n"
                                f"optimize_memory_usage : {agent_params._ddpg.OPTIMIZE_MEMORY_USAGE}\n"
                                f"tensorboard_log    : {agent_params._ddpg.TENSORBOARD_LOG}\n"
                                f"create_eval_env    : {agent_params._ddpg.CREATE_EVAL_ENV}\n"
                                f"policy_kwargs      : {agent_params._ddpg.POLICY_KWARGS}\n"
                                f"verbose            : {agent_params._ddpg.VERBOSE}\n"
                                f"device             : {agent_params._ddpg.DEVICE}\n"
                                f"init_setup_model   : {agent_params._ddpg.INIT_SETUP_MODEL}\n"
                                f"TRAINING_TIMESTEPS : {agent_params._ddpg.TRAINING_TIMESTEPS}\n")
        elif agent_ == "a2c":
            with open(txtfile_path, "a") as text_file:
                text_file.write(f"policy             : {agent_params._a2c.POLICY}\n"
                                f"ent_coef           : {agent_params._a2c.ENT_COEF}\n"
                                f"vf_coef            : {agent_params._a2c.VF_COEF}\n"
                                f"learning_rate      : {agent_params._a2c.LEARNING_RATE}\n"
                                f"gamma              : {agent_params._a2c.GAMMA}\n"
                                f"gae_lambda         : {agent_params._a2c.GAE_LAMBDA}\n"
                                f"max_grad_norm      : {agent_params._a2c.MAX_GRAD_NORM}\n"
                                f"rms_prop_eps       : {agent_params._a2c.RMS_PROP_EPS}\n"
                                f"use_rms_prop       : {agent_params._a2c.USE_RMS_PROP}\n"
                                f"n_steps            : {agent_params._a2c.N_STEPS}\n"
                                f"use_sde            : {agent_params._a2c.USE_SDE}\n"
                                f"sde_sample_freq    : {agent_params._a2c.SDE_SAMPLE_FREQ}\n"
                                f"normalize_advantage: {agent_params._a2c.NORMALIZE_ADVANTAGE}\n"
                                f"tensorboard_log    : {agent_params._a2c.TENSORBOARD_LOG}\n"
                                f"create_eval_env    : {agent_params._a2c.CREATE_EVAL_ENV}\n"
                                f"policy_kwargs      : {agent_params._a2c.POLICY_KWARGS}\n"
                                f"verbose            : {agent_params._a2c.VERBOSE}\n"
                                f"device             : {agent_params._a2c.DEVICE}\n"
                                f"init_setup_model   : {agent_params._a2c.INIT_SETUP_MODEL}\n"
                                f"TRAINING_TIMESTEPS : {agent_params._a2c.TRAINING_TIMESTEPS}\n")
    return None


def run_model_pipeline(data,
              results_dir,
              trained_dir,
              TB_log_dir,
              stock_dim,
              n_features,
              shape_observation_space,
              #unique_trade_dates_validation
              ) -> None:
    """
    Runs the whole setup: training, validation, trading
    """
    # CHOOSE MODEL TO RUN
    # -------
    # if we don't run an ensemble strategy, we run the single agent strategy
    if settings.STRATEGY_MODE.find("ens") == -1:
        #logging.warning(f"(RUN) STRATEGY_MODE: {settings.STRATEGY_MODE}.")
        # Single Agent Only; call function in "models.py"
        run_model(df=data,
                  results_dir=results_dir,
                  trained_dir=trained_dir,
                  TB_log_dir=TB_log_dir,
                  stock_dim=stock_dim,
                  n_features=n_features,
                  shape_observation_space=shape_observation_space,
                  #unique_trade_dates_validation=unique_trade_dates_validation
                    )
    elif settings.STRATEGY_MODE.find("ens") != -1:
    #logging.warning(f"(RUN) STRATEGY_MODE: {settings.STRATEGY_MODE}.")
        #run_ensemble(df=data,
         #            REBALANCE_WINDOW=settings.REBALANCE_WINDOW,
         #            validation_window=settings.VALIDATION_WINDOW,
         #            strategy_mode=settings.STRATEGY_MODE,
         #            crisis_measure=crisis_settings.CRISIS_MEASURE)
        pass
    else:
        logging.warning(f"(RUN) STRATEGY_MODE is not specified ({settings.STRATEGY_MODE}).")
        pass
