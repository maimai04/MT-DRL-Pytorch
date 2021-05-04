import pandas as pd
import numpy as np
import time
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from preprocessing.preprocessors import *
# config
from config.config import *
from config.config import settings, crisis_settings, paths, env_params, dataprep_settings
# model
from model.models_pipeline import *
import os

######################################################
##   DEFINING FUNCTIONS USED IN the run__.py file   ##
######################################################

def create_dirs(mode="run_dir", # or "seed_dir"
                results_dir="",
                trained_dir=""
                ) -> list: # todo: create support_functions and move there
    """
    Creates directories for results and the trained model, based on paths specified in the config.py file.
    @return: results_dir, trained_dir
    """
    from config.config import settings, crisis_settings, paths, env_params, dataprep_settings

    if mode == "run_dir":
        ### RESULTY DIRECTORY
        # creating results directory for the current run folders (one run/folder for each seed in this directory)
        results_dir = os.path.join(paths.RESULTS_PATH,
                                   f"{settings.NOW}_{settings.STRATEGY_MODE}_"
                                   f"{crisis_settings.CNAME}_{dataprep_settings.FEATURES_MODE}")
        os.makedirs(results_dir) # os.makedirs() method creates all unavailable/missing directories under the specified path

        ### TRAINED MODEL DIRECTORY (saving the trained DRL models)
        trained_dir = os.path.join(paths.TRAINED_MODELS_PATH, f"{settings.NOW}_{settings.STRATEGY_MODE}_"
                                   f"{crisis_settings.CNAME}_{dataprep_settings.FEATURES_MODE}")
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
                          seeds_list,
                          logsave_path
                          ) -> None:
    # todo: create support_functions and move there
    """

    Writes all configurations and related parameters into the config_log.txt file.

    @param results_dir: the directory where the results for the current run are saved
    @param trained_dir: the directory where the model of the current run is saved
    @return: None

    """
    from config.config import settings, crisis_settings, paths, env_params
    txtfile_path = os.path.join(logsave_path, "configurations.txt")

    with open(txtfile_path, "w") as text_file:
        text_file.write("------------------------------------\n"
                        "CONFIGURATION SETTINGS AND VARIABLES\n"
                        "------------------------------------\n"
                        "------------------------------------\n"
                        "SETTINGS\n"
                        "------------------------------------\n"
                        f"NOW                  : {settings.NOW}\n"
                        f"SEEDS LIST           : {seeds_list}\n"
                        f"STRATEGY_MODE        : {settings.STRATEGY_MODE}\n"
                        f"AGENTS_LIST          : {settings.AGENTS_LIST}\n"
                        f"REBALANCE_WINDOW     : {settings.REBALANCE_WINDOW}\n"
                        f"VALIDATION_WINDOW    : {settings.VALIDATION_WINDOW}\n"
                        f"STARTDATE_TRAIN      : {settings.STARTDATE_TRAIN}\n"
                        f"ENDDATE_TRAIN        : {settings.ENDDATE_TRAIN}\n"
                        f"STARTDATE_VALIDATION : {settings.STARTDATE_VALIDATION}\n"
                        f"ENDDATE_VALIDATION   : {settings.ENDDATE_VALIDATION}\n"
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
                        f"PREPROCESS_ANEW  : {dataprep_settings.PREPROCESS_ANEW}\n"
                        f"FEATURES_LIST    : {dataprep_settings.FEATURES_LIST}\n"
                        f"FEATURES_MODE    : {dataprep_settings.FEATURES_MODE}\n"
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
                text_file.write(f"policy             : {agent_params._ppo.policy}\n"
                                f"ent_coef           : {agent_params._ppo.ent_coef}\n"
                                f"default parameters (unchanged from default as given by stable-baselines):"
                                f"gamma              : {agent_params._ppo.gamma}\n"
                                f"learning_rate      : {agent_params._ppo.learning_rate}\n"
                                f"n_steps            : {agent_params._ppo.n_steps}\n"
                                f"batch_size         : Optional[int] = 64\n"
                                f"n_epochs           : {agent_params._ppo.n_epochs}\n"
                                f"gae_lambda         : {agent_params._ppo.gae_lambda}\n"
                                f"clip_range_vf      : {agent_params._ppo.clip_range_vf}\n"
                                f"clip_range         : {agent_params._ppo.clip_range}\n"
                                f"vf_coef            : {agent_params._ppo.vf_coef}\n"
                                f"max_grad_norm      : {agent_params._ppo.max_grad_norm}\n"
                                f"use_sde            : {agent_params._ppo.use_sde}\n"
                                f"sde_sample_freq    : {agent_params._ppo.sde_sample_freq}\n"
                                f"target_kl          : {agent_params._ppo.target_kl}\n"
                                f"tensorboard_log    : {agent_params._ppo.tensorboard_log}\n"
                                f"create_eval_env    : {agent_params._ppo.create_eval_env}\n"
                                f"policy_kwargs      : {agent_params._ppo.policy_kwargs}\n"
                                f"verbose            : {agent_params._ppo.verbose}\n"
                                f"device             : {agent_params._ppo.device}\n"
                                f"init_setup_model   : {agent_params._ppo.init_setup_model}\n"
                                f"TRAINING_TIMESTEPS : {agent_params._ppo.TRAINING_TIMESTEPS}\n")
        elif agent_ == "ddpg":
            with open(txtfile_path, "a") as text_file:
                text_file.write(f"policy             : {agent_params._ddpg.policy}\n"
                                f"action_noise       : {agent_params._ddpg.action_noise}\n"
                                f"default parameters (unchanged from default as given by stable-baselines):"
                                f"gamma              : {agent_params._ddpg.gamma}\n"
                                f"learning_rate      : {agent_params._ddpg.learning_rate}\n"
                                f"buffer_size        : {agent_params._ddpg.buffer_size}\n"
                                f"learning_starts    : {agent_params._ddpg.learning_starts}\n"
                                f"batch_size         : {agent_params._ddpg.batch_size}\n"
                                f"tau                : {agent_params._ddpg.tau}\n"
                                f"gradient_steps     : {agent_params._ddpg.gradient_steps}\n"
                                f"optimize_memory_usage : {agent_params._ddpg.optimize_memory_usage}\n"
                                f"tensorboard_log    : {agent_params._ddpg.tensorboard_log}\n"
                                f"create_eval_env    : {agent_params._ddpg.create_eval_env}\n"
                                f"policy_kwargs      : {agent_params._ddpg.policy_kwargs}\n"
                                f"verbose            : {agent_params._ddpg.verbose}\n"
                                f"device             : {agent_params._ddpg.device}\n"
                                f"init_setup_model   : {agent_params._ddpg.init_setup_model}\n"
                                f"TRAINING_TIMESTEPS : {agent_params._ddpg.TRAINING_TIMESTEPS}\n")
        elif agent_ == "a2c":
            with open(txtfile_path, "a") as text_file:
                text_file.write(f"policy             : {agent_params._a2c.policy}\n"
                                f"ent_coef           : {agent_params._a2c.ent_coef}\n"
                                f"vf_coef            : {agent_params._a2c.vf_coef}\n"
                                f"learning_rate      : {agent_params._a2c.learning_rate}\n"
                                f"gamma              : {agent_params._a2c.gamma}\n"
                                f"gae_lambda         : {agent_params._a2c.gae_lambda}\n"
                                f"max_grad_norm      : {agent_params._a2c.max_grad_norm}\n"
                                f"rms_prop_eps       : {agent_params._a2c.rms_prop_eps}\n"
                                f"use_rms_prop       : {agent_params._a2c.use_rms_prop}\n"
                                f"n_steps            : {agent_params._a2c.n_steps}\n"
                                f"use_sde            : {agent_params._a2c.use_sde}\n"
                                f"sde_sample_freq    : {agent_params._a2c.sde_sample_freq}\n"
                                f"normalize_advantage: {agent_params._a2c.normalize_advantage}\n"
                                f"tensorboard_log    : {agent_params._a2c.tensorboard_log}\n"
                                f"create_eval_env    : {agent_params._a2c.create_eval_env}\n"
                                f"policy_kwargs      : {agent_params._a2c.policy_kwargs}\n"
                                f"verbose            : {agent_params._a2c.verbose}\n"
                                f"device             : {agent_params._a2c.device}\n"
                                f"init_setup_model   : {agent_params._a2c.init_setup_model}\n"
                                f"TRAINING_TIMESTEPS : {agent_params._a2c.TRAINING_TIMESTEPS}\n")
    return None

def data_handling(# define if you want to pre-process anew or not (usually done in config.py file) # todo: create support_functions and move there
                preprocess_anew=dataprep_settings.PREPROCESS_ANEW,
                # PARAMETERS FOR DATA IMPORT (EITHER PRE-PROCESSED DF OR RAW DF)
                preprocessed_data_file=None, #paths.PREPROCESSED_DATA_FILE,
                save_path=None,  # paths.PREPROCESSED_DATA_PATH

                # BASE PARAMS FOR LOADING THE RAW DATA SET for preprocessing pipeline - with load_dataset()
                raw_data_file=None, #paths.RAW_DATA_FILE,
                col_subset=None,#dataprep_settings.RAW_DF_COLS_SUBSET,
                date_subset=None,#"datadate",
                date_subset_startdate=None, #settings.STARTDATE_TRAIN,

                # PASSING NAMES OF OPTIONAL FUNCTIONS TO BE USED in preprocessing pipeline
                calculate_price_volume_func="calculate_price_volume_WhartonData",
                add_technical_indicator_func="add_technical_indicator_with_StockStats",
                add_other_features_func=None,  # "add_other_features",
                add_ANN_features_func=None,
                add_crisis_measure_func="add_crisis_measure",

                # PASSING PARAMETERS FOR EACH OPTIONAL FUNCTION
                # params for calculate_price_function()
                calculate_price_volume_func_params={"new_cols_subset": dataprep_settings.NEW_COLS_SUBSET,
                                                    "target_subset": None},
                # params for add_technical_indicator_func
                add_technical_indicator_func_params={"technical_indicators_list": ["macd", "rsi_30", "cci_30", "dx_30"]},
                # params for adding other features (e.g. volatility)
                add_other_features_func_params={"feature": "returns_volatility",
                                                "window_days": 7},
                # params for adding ANN-created features
                add_ANN_features_func_params={},
                # params for add_crisis_measure_func
                add_crisis_measure_func_params={"crisis_measure": crisis_settings.CRISIS_MEASURE},
                ):
    """
    Handles the data set to be use dfor the modelind.
    If preprocess_anew = False, loads and returns the already-preprocessed dataset.
    If preprocess_anew = True, calls the data_preprocessing_pipeline function from preprocessors.py, using the
    preprocessing specifications handed to the function (config.txt: dataprep_settings)

    @param df: raw or preprocessed
    @return: (df) pandas dataframe, fully prepared and can be used directly for the modeling without any further steps.

    Note: we can either use a raw data set and pre-process it according to the settings given in config.py
    and preprocessors.py, or we can use an already pre-processed data set directly.
    There is no option choosing a raw data set without pre-processing, because this could lead to errors
    (e.g. missing values, format...), but we can just choose to pre-process the raw dataset only w.r.t. the absolute
    minimum / basics needed (specify in config.py or manually in this function).

    """
    # if we don't want to pre-process the data (settings in config.py), we need to import an existing per-processed df
    if preprocess_anew == False:
        # get name of preprocessed data file based on features mode (fm), set in config.py file
        # unless we have explicitly defined the file to be used in the functions parameters

        if preprocessed_data_file is None:
            preprocessed_data_file = os.path.join(paths.PREPROCESSED_DATA_PATH, f"data_{dataprep_settings.DATASET_CODE}_" 
                                                            f"{dataprep_settings.DATABASE}_"
                                                            f"{dataprep_settings.FEATURES_MODE}.csv")
            logging.warning(f"preprocessed_data_file: , {preprocessed_data_file}")
        elif preprocessed_data_file is not None:
            pass

        if os.path.exists(preprocessed_data_file):
            logging.warning(f"Data handling: Using existing pre-processed data: {preprocessed_data_file}")
            # data = pd.read_csv(path_of_preprocessed_file, index_col=0) # todo: using load dataset from preprocessors
            data = load_dataset(file_path=preprocessed_data_file,
                                    col_subset=None,
                                    date_subset=None,
                                    date_subset_startdate=None,
                                    )
            data = data.sort_values([dataprep_settings.DATE_COLUMN, dataprep_settings.ASSET_NAME_COLUMN])
            data.index = data[dataprep_settings.DATE_COLUMN].factorize()[0]
        else:
            raise Exception("Chosen option -using already pre-processed data set-, "
                            "but no pre-processed data existent under given path:\n"
                            f"{preprocessed_data_file}")
    # if we want to pre-process a raw data set (settings in config.py),
    # we need to import the raw data set and apply preprocessors from preprocessors.py
    elif preprocess_anew == True:
        if os.path.exists(raw_data_file):
            logging.warning(f"Data handling: Preprocessing the raw data set; {raw_data_file}")
            # call function preprocess_data from preprocessors.py, specifying level of pre-processing
            data = data_preprocessing_pipeline(# BASE PARAMS FOR LOADING THE DATA SET - with load_dataset()
                                                raw_data_file=raw_data_file,
                                                col_subset=dataprep_settings.RAW_DF_COLS_SUBSET,
                                                date_subset="datadate",
                                                date_subset_startdate=settings.STARTDATE_TRAIN,

                                                # PASSING NAMES OF OPTIONAL FUNCTIONS TO BE USED
                                                calculate_price_volume_func=calculate_price_volume_func,
                                                add_technical_indicator_func=add_technical_indicator_func,
                                                add_other_features_func=add_other_features_func,  # "add_other_features",
                                                add_ANN_features_func=add_ANN_features_func,
                                                add_crisis_measure_func=add_crisis_measure_func,

                                                # PASSING PARAMETERS FOR EACH OPTIONAL FUNCTION
                                                calculate_price_volume_func_params=calculate_price_volume_func_params,
                                                add_technical_indicator_func_params=add_technical_indicator_func_params,
                                                add_other_features_func_params=add_other_features_func_params,
                                                add_ANN_features_func_params=add_ANN_features_func_params,
                                                add_crisis_measure_func_params=add_crisis_measure_func_params)

            save_df_path = os.path.join(save_path,
                                        f"data_{dataprep_settings.DATASET_CODE}_"
                                        f"{dataprep_settings.DATABASE}_"
                                        f"{dataprep_settings.FEATURES_MODE}.csv")
            data.to_csv(save_df_path)
            data = data.sort_values([dataprep_settings.DATE_COLUMN, dataprep_settings.ASSET_NAME_COLUMN])
            data.index = data[dataprep_settings.DATE_COLUMN].factorize()[0]
        else:
            raise Exception("Chosen option -pre-processing raw data set-, "
                            "but no raw file existent under given path:\n"
                            f"{raw_data_file}")

    logging.warning("FINAL INPUT DATAFRAME")
    logging.warning("---------------------------------------------------------")
    logging.warning(data.head())
    logging.warning(f"Shape of Dataframe (rows, columns)     : {data.shape}")
    logging.warning(f"Size of Dataframe (total n. of elemets): {data.shape}\n ")

    return data


def run_model_pipeline(data,
              results_dir,
              trained_dir,
              stock_dim,
              n_features,
              shape_observation_space,
              unique_trade_dates_validation
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
        run_model2(df=data,
                  results_dir=results_dir,
                  trained_dir=trained_dir,
                  stock_dim=stock_dim,
                  n_features=n_features,
                  shape_observation_space=shape_observation_space,
                  unique_trade_dates_validation=unique_trade_dates_validation)
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
