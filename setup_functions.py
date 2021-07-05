import os
import pandas as pd
from model.run_pipeline import *
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
    results_path = paths.RESULTS_PATH
    trained_models_path = paths.TRAINED_MODELS_PATH

    if mode == "run_dir":
        ### RESULTY DIRECTORY
        # creating results directory for the current run folders (one run/folder for each seed in this directory)
        results_dir = os.path.join(results_path,
                                   f"{settings.NOW}_{settings.STRATEGY_MODE}_"
                                   f"{data_settings.FEATURES_MODE}"
                                   f"_{settings.RUN_MODE}")
        os.makedirs(results_dir) # os.makedirs() method creates all unavailable/missing directories under the specified path

        ### TRAINED MODEL DIRECTORY (saving the trained DRL models)
        trained_dir = os.path.join(trained_models_path, f"{settings.NOW}"
                                                              f"_{settings.STRATEGY_MODE}"
                                                              f"_rew_{settings.REWARD_MEASURE}"
                                                              f"_{data_settings.FEATURES_MODE}"
                                                              f"_{settings.RUN_MODE}")
        os.makedirs(trained_dir)
        return [results_dir, trained_dir]

    if mode == "seed_dir":
        ### RESULTY SUBDIRECTORY
        # creating results sub-directory, one for each seed (for which the algorithm is run) within one run
        results_subdir = os.path.join(results_dir, f"agentSeed{settings.SEED}")
        os.makedirs(results_subdir)
        ### TRAINED MODEL SUBDIRECTORY
        trained_subdir = os.path.join(trained_dir, f"agentSeed{settings.SEED}")
        os.makedirs(trained_subdir)

        # creating sub-sub-directories for the actual results folders (e.g. portfolio_value etc.)
        # where the resulting .csv files are saved during the run
        # the names of the sub-sub-directories are defined in the config file under paths.SUBSUBDIR_NAMES
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
                        f"REWARD_MEASURE       : {settings.REWARD_MEASURE}\n"
                        f"RETRAIN_DATA         : {settings.RETRAIN_DATA}\n"
                        f"RUN_MODE             : {settings.RUN_MODE}\n"
                        f"STARTDATE_TRAIN      : {settings.STARTDATE_TRAIN}\n"
                        f"ENDDATE_TRAIN        : {settings.ENDDATE_TRAIN}\n"
                        f"ROLL_WINDOW          : {settings.ROLL_WINDOW}\n"

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
                        f"FEATURES_LIST            : {data_settings.FEATURES_LIST}\n"
                        f"FEATURES_MODE            : {data_settings.FEATURES_MODE}\n"
                        "------------------------------------\n"
                        f"PATHS AND DIRECTORIES\n"
                        "------------------------------------\n"
                        f"DATA_PATH                : {paths.DATA_PATH}\n"
                        f"RAW_DATA_PATH            : {paths.RAW_DATA_PATH}\n"
                        f"PREPROCESSED_DATA_PATH   : {paths.PREPROCESSED_DATA_PATH}\n"
                        f"TRAINED_MODELS_PATH      : {paths.TRAINED_MODELS_PATH}\n"
                        f"RESULTS_PATH             : {paths.RESULTS_PATH}\n"
                        f"SUBDIR_NAMES             : {paths.SUBSUBDIR_NAMES}\n"
                        #f"RAW_DATA_FILE            : {paths.RAW_DATA_FILE}\n"
                        f"PREPROCESSED_DATA_FILE   : {paths.PREPROCESSED_DATA_FILE}\n"
                        f"RESULTS_DIR              : {results_subdir}\n"
                        f"TRAINED_MODEL_DIR        : {trained_subdir}\n")  # trained_model_dir
        if settings.STRATEGY_MODE == "ppoCustomBase":
            with open(txtfile_path, "a") as text_file:
                text_file.write(f"NET_VERSION                   : {agent_params.ppoCustomBase.NET_VERSION}\n"
                                f"BATCH_SIZE                    : {agent_params.ppoCustomBase.BATCH_SIZE}\n"
                                f"NUM_EPOCHS                    : {agent_params.ppoCustomBase.NUM_EPOCHS}\n"
                                f"OPTIMIZER                     : {agent_params.ppoCustomBase.OPTIMIZER}\n"
                                f"OPTIMIZER_LEARNING_RATE       : {agent_params.ppoCustomBase.OPTIMIZER_LEARNING_RATE}\n"
                                f"GAMMA                         : {agent_params.ppoCustomBase.GAMMA}\n"
                                f"GAE_LAMBDA                    : {agent_params.ppoCustomBase.GAE_LAMBDA}\n"
                                f"CLIP_EPSILON                  : {agent_params.ppoCustomBase.CLIP_EPSILON}\n"
                                f"MAX_KL_VALUE                  : {agent_params.ppoCustomBase.MAX_KL_VALUE}\n"
                                f"CRITIC_LOSS_COEF              : {agent_params.ppoCustomBase.CRITIC_LOSS_COEF}\n"
                                f"ENTROPY_LOSS_COEF             : {agent_params.ppoCustomBase.ENTROPY_LOSS_COEF}\n"
                                f"MAX_GRADIENT_NORMALIZATION    : {agent_params.ppoCustomBase.MAX_GRADIENT_NORMALIZATION}\n"
                                f"TOTAL_TIMESTEPS_TO_COLLECT    : {agent_params.ppoCustomBase.TOTAL_TIMESTEPS_TO_COLLECT}\n"
                                f"TOTAL_TIMESTEPS_TO_TRAIN      : {agent_params.ppoCustomBase.TOTAL_TIMESTEPS_TO_TRAIN}\n"
                                )
        if settings.STRATEGY_MODE == "ppo":
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


def get_data_params(final_df: pd.DataFrame,
                    feature_cols=data_settings.FEATURES_LIST,
                    asset_name_column="tic",
                    ) -> list:
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

    return [n_individual_assets, n_features]

