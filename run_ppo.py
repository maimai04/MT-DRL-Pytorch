import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv
import logging
from preprocessing.preprocessors import *
# config
from config.config import *
from config.config import settings, crisis_settings, paths, env_params, dataprep_settings
# model
from model.models import *
import os

"""
run only PPO
everything else c.p.

additional changes from BCAP:
- added comments
- added variable which_run, to save subsequent runs in corresponding folders.
- 
"""

############################
##   DEFINING FUNCTIONS   ##
############################

def create_dirs() -> list: # todo: create support_functions and move there
    """
    Creates directories for results and the trained model, based on paths specified in the config.py file.
    @return: results_dir, trained_dir
    """
    from config.config import settings, crisis_settings, paths, env_params, dataprep_settings
    # ---------------LEAVE---------------
    results_dir = os.path.join(paths.RESULTS_PATH,
                               f"{settings.NOW}_agentSeed{settings.SEED_PPO}_{settings.STRATEGY_MODE}_{crisis_settings.CNAME}_{dataprep_settings.FEATURES_MODE}")
    trained_dir = os.path.join(paths.TRAINED_MODELS_PATH,
                               f"{settings.NOW}_agentSeed{settings.SEED_PPO}_{settings.STRATEGY_MODE}_{crisis_settings.CNAME}_{dataprep_settings.FEATURES_MODE}")
    os.makedirs(results_dir)
    os.makedirs(trained_dir)
    # os.makedirs() method will create all unavailable/missing directory in the specified path.

    # for each key (=directory name) in paths.SUBDIR_NAMES (see config.py file),
    # create a subdirectory path and based on this path, create a subdirectory in the results folder
    # there, we will later save our results into (e.g. account memory, rewards, costs etc.)
    for dirname in paths.SUBDIR_NAMES.keys():
        subdir_path = os.path.join(results_dir, dirname)
        os.makedirs(subdir_path)
    del subdir_path

    return [results_dir, trained_dir]


def config_logging_to_txt(results_dir, trained_dir) -> None: # todo: create support_functions and move there
    """

    Writes all configurations and related parameters into the config_log.txt file.

    @param results_dir: the directory where the results for the current run are saved
    @param trained_dir: the directory where the model of the current run is saved
    @return: None

    """
    from config.config import settings, crisis_settings, paths, env_params
    subdir_path = os.path.join(results_dir, "_LOGGINGS")
    os.makedirs(subdir_path)
    logging.basicConfig(filename=os.path.join(subdir_path, "config_log"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        # asctime = time, msecs, name = name of who runned it (?), levelname (e.g. DEBUG, INFO => verbosity), message
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logging.info("------------------------------------")
    logging.info("CONFIGURATION SETTINGS AND VARIABLES")
    logging.info("------------------------------------")

    logging.info("\nSETTINGS")
    logging.info("------------------------------------")
    logging.info(f"STRATEGY_MODE: {settings.STRATEGY_MODE}")
    logging.info(f"REBALANCE_WINDOW: {settings.REBALANCE_WINDOW}")
    logging.info(f"VALIDATION_WINDOW: {settings.VALIDATION_WINDOW}")
    logging.info(f"SEED_PPO: {settings.SEED_PPO}")
    logging.info(f"SEED_ENV: {settings.SEED_ENV}")
    logging.info(f"NOW: {settings.NOW}")

    logging.info("\nDATA PREPARATION SETTINGS")
    logging.info("------------------------------------")
    logging.info(f"PREPROCESS_ANEW: {dataprep_settings.PREPROCESS_ANEW}")
    logging.info(f"FEATURES_LIST: {dataprep_settings.FEATURES_LIST}")
    logging.info(f"FEATURES_MODE: {dataprep_settings.FEATURES_MODE}")

    logging.info("\nCRISIS SETTINGS")
    logging.info("------------------------------------")
    logging.info(f"CRISIS_MEASURE: {crisis_settings.CRISIS_MEASURE}")

    logging.info("\nPATHS AND DIRECTORIES")
    logging.info("------------------------------------")
    logging.info(f"DATA_PATH: {paths.DATA_PATH}")
    logging.info(f"RAW_DATA_PATH: {paths.RAW_DATA_PATH}")
    logging.info(f"PREPROCESSED_DATA_PATH: {paths.PREPROCESSED_DATA_PATH}")
    logging.info(f"TRAINED_MODELS_PATH: {paths.TRAINED_MODELS_PATH}")
    logging.info(f"RESULTS_PATH: {paths.RESULTS_PATH}")
    logging.info(f"SUBDIR_NAMES: {paths.SUBDIR_NAMES}")
    logging.info(f"TESTING_DATA_FILE: {paths.TESTING_DATA_FILE}")  # todo: where is this file?
    logging.info(f"RAW_DATA_FILE: {paths.RAW_DATA_FILE}")
    logging.info(f"PREPROCESSED_DATA_FILE: {paths.PREPROCESSED_DATA_FILE}")
    logging.info(f"RESULTS_DIR: {results_dir}")
    logging.info(f"TRAINED_MODEL_DIR: {trained_dir}")  # trained_model_dir

    logging.info("\nENVIRONMENT VARIABLES")
    logging.info("------------------------------------")
    logging.info(f"HMAX_NORMALIZE: {env_params.HMAX_NORMALIZE}")
    logging.info(f"INITIAL_CASH_BALANCE: {env_params.INITIAL_CASH_BALANCE}")
    logging.info(f"TRANSACTION_FEE_PERCENT: {env_params.TRANSACTION_FEE_PERCENT}")
    logging.info(f"REWARD_SCALING: {env_params.REWARD_SCALING}")
    #logging.info(f"SHAPE_OBSERVATION_SPACE: {env_params.SHAPE_OBSERVATION_SPACE}")
    #logging.info(f"STOCK_DIM: {env_params.STOCK_DIM}")

    del subdir_path
    return None

def data_handling(# define if you want to pre-process anew or not (usually done in comfig.py file) # todo: create support_functions and move there
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
                add_other_features_func_params={},
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
        # if there exists no
        if os.path.exists(preprocessed_data_file):
            print(f"Using existing pre-processed data; {preprocessed_data_file}")
            # data = pd.read_csv(path_of_preprocessed_file, index_col=0) # todo: using load dataset from preprocessors
            data = load_dataset(file_path=preprocessed_data_file,
                                col_subset=None,
                                date_subset=None,
                                date_subset_startdate=None,
                                )
        else:
            raise Exception("Chosen option -using already pre-processed data set-, "
                            "but no pre-processed data existent under given path:\n"
                            f"{preprocessed_data_file}")
    # if we want to pre-process a raw data set (settings in config.py),
    # we need to import the raw data set and apply preprocessors from preprocessors.py
    elif preprocess_anew == True:
        if os.path.exists(raw_data_file):
            print(f"Preprocessing the raw data set; {raw_data_file}")
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

            save_df_path = os.path.join(save_path, f"data_preprocessed_on_{settings.NOW}")
            data.to_csv(save_df_path)
        else:
            raise Exception("Chosen option -pre-processing raw data set-, "
                            "but no raw file existent under given path:\n"
                            f"{raw_data_file}")

    print("\nFINAL INPUT DATAFRAME")
    print("---------------------------------------------------------\n")
    print(data.head())
    print(f"\nShape of Dataframe (rows, columns)     : {data.shape}")
    print(f"Size of Dataframe (total n. of elemets): {data.shape}")

    return data


def run_model(data,
              results_dir,
              trained_dir,
              ) -> None:
    """
    Runs the whole setup: training, validation, trading
    """
    # MODEL
    # -------
    if settings.STRATEGY_MODE == "ppo":
        print(f"(RUN) STRATEGY_MODE: {settings.STRATEGY_MODE}.")
        # PPO Only; call function in "models.py"
        run_single_agent(df=data,
                         results_dir=results_dir,
                         trained_dir=trained_dir)
        pass
    elif settings.STRATEGY_MODE == "ensemble":
        print(f"(RUN) STRATEGY_MODE: {settings.STRATEGY_MODE}.")
        #run_ensemble(df=data,
         #            REBALANCE_WINDOW=settings.REBALANCE_WINDOW,
         #            validation_window=settings.VALIDATION_WINDOW,
         #            strategy_mode=settings.STRATEGY_MODE,
         #            crisis_measure=crisis_settings.CRISIS_MEASURE)
        pass
    else:
        print(f"(RUN) STRATEGY_MODE is not specified ({settings.STRATEGY_MODE}).")
        pass

############################
##     MAIN CONDITION     ##
############################

if __name__ == "__main__":
    #### SETUP
    ####------------------------------
    # call function create_dirs() specified above to create the directory for saving results and the trained model
    # based on the paths/parameters specified in the config.py file
    results_dir, trained_dir = create_dirs()

    # call function to write all configurations / related parameters into the config_log.txt file in folder _LOGGING
    # so later this file can be consulted to uniquely identify each run and its configurations
    config_logging_to_txt(results_dir, trained_dir)

    #### LOAD / LOAD & PREPROCESS DATA
    ####-------------------------------
    # call function to get the preprocessed data / get the raw data and preprocess it, depending on params in config.py
    data = data_handling( # simplify, make more readable and move mostly into config file
                    preprocess_anew=dataprep_settings.PREPROCESS_ANEW,
                    preprocessed_data_file=paths.PREPROCESSED_DATA_FILE,
                    save_path=paths.PREPROCESSED_DATA_PATH,
                    # BASE PARAMS FOR LOADING THE RAW DATA SET for preprocessing pipeline - with load_dataset()
                    raw_data_file=paths.RAW_DATA_FILE,
                    col_subset=dataprep_settings.RAW_DF_COLS_SUBSET,
                    date_subset="datadate",
                    date_subset_startdate=settings.STARTDATE_TRAIN,
                    # PASSING NAMES OF OPTIONAL FUNCTIONS TO BE USED in preprocessing pipeline
                    calculate_price_volume_func="calculate_price_volume_WhartonData",
                    add_technical_indicator_func="add_technical_indicator_with_StockStats",
                    add_other_features_func=None,  # "add_other_features",
                    add_ANN_features_func=None,
                    add_crisis_measure_func="add_crisis_measure",
                    # PASSING PARAMETERS FOR EACH OPTIONAL FUNCTION
                    calculate_price_volume_func_params={"new_cols_subset": dataprep_settings.NEW_COLS_SUBSET,
                                                          "target_subset": None},
                    add_technical_indicator_func_params={"technical_indicators_list": dataprep_settings.TECH_INDICATORS},
                    add_other_features_func_params={},
                    add_ANN_features_func_params={},
                    add_crisis_measure_func_params={"crisis_measure": crisis_settings.CRISIS_MEASURE},
                    )

    #### RUN MODEL SETUP
    ####-------------------------------
    # call run_model function to run the actual DRL algorithm

    run_model(data=data,
              results_dir=results_dir,
              trained_dir=trained_dir,
              )
