import logging
from preprocessing.preprocessors import *
from config.config import *
from config.config import settings, crisis_settings, paths, env_params, dataprep_settings
from setup_functions import *
from model.models import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed

"""
run only PPO
everything else c.p.

additional changes from BCAP:
- added comments
- added variable which_run, to save subsequent runs in corresponding folders.
- 
"""

############################
##     MAIN CONDITION     ##
############################

if __name__ == "__main__":

    seeds_list = [223445, 80923, 11112, 23, 5]

    results_dir, trained_dir = create_dirs(mode="run_dir")
    logsave_path = os.path.join(results_dir, "_LOGGINGS")
    os.makedirs(logsave_path)

    # call function to write all configurations / related parameters into the config_log.txt file in folder _LOGGING
    # so later this file can be consulted to uniquely identify each run and its configurations
    config_logging_to_txt(results_dir, trained_dir, seeds_list=seeds_list, logsave_path=logsave_path)
    logging.basicConfig(filename=os.path.join(logsave_path, "run_log"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        # asctime = time, msecs, name = name of who runned it (?), levelname (e.g. DEBUG, INFO => verbosity), message
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    #### LOAD / LOAD & PREPROCESS DATA
    ####-------------------------------
    # call function to get the preprocessed data / get the raw data and preprocess it, depending on params in config.py
    data = data_handling(  # simplify, make more readable and move mostly into config file
        preprocess_anew=dataprep_settings.PREPROCESS_ANEW,
        preprocessed_data_file=None,  # paths.PREPROCESSED_DATA_FILE,
        save_path=paths.PREPROCESSED_DATA_PATH,
        # BASE PARAMS FOR LOADING THE RAW DATA SET for preprocessing pipeline - with load_dataset()
        raw_data_file=paths.RAW_DATA_FILE,
        col_subset=dataprep_settings.RAW_DF_COLS_SUBSET,
        date_subset="datadate",
        date_subset_startdate=settings.STARTDATE_TRAIN,
        # PASSING NAMES OF OPTIONAL FUNCTIONS TO BE USED in preprocessing pipeline
        calculate_price_volume_func="calculate_price_volume_WhartonData",
        add_technical_indicator_func="add_technical_indicator_with_StockStats",
        # add_other_features_func=None,  # "add_other_features"
        add_other_features_func="add_other_features",
        add_ANN_features_func=None,
        add_crisis_measure_func="add_crisis_measure",
        # PASSING PARAMETERS FOR EACH OPTIONAL FUNCTION
        calculate_price_volume_func_params={"new_cols_subset": dataprep_settings.NEW_COLS_SUBSET,
                                            "target_subset": None},
        add_technical_indicator_func_params={"technical_indicators_list": dataprep_settings.TECH_INDICATORS},
        add_other_features_func_params={"feature": "returns_volatility",
                                        "window_days": 7},
        add_ANN_features_func_params={},
        add_crisis_measure_func_params={"crisis_measure": crisis_settings.CRISIS_MEASURE},
    )

    # SETUP
    # -------
    # get parameters about dataframe shape
    stock_dim, n_features, unique_trade_dates_validation = \
        get_data_params(final_df=data,
                        asset_name_column=dataprep_settings.ASSET_NAME_COLUMN,
                        feature_cols=dataprep_settings.ALL_USED_COLUMNS,
                        date_column=dataprep_settings.DATE_COLUMN,
                        base_cols=dataprep_settings.BASE_DF_COLS,
                        startdate_validation=settings.STARTDATE_VALIDATION,
                        enddate_validation=settings.ENDDATE_VALIDATION,
                        )
    # save unique trade dates to csv
    utd = pd.DataFrame(unique_trade_dates_validation)
    utd.to_csv(os.path.join(results_dir, "unique_trade_dates_validation.csv"), index=False)
    del utd

    # Shape = [Current Balance]+[prices 1-30]+[owned shares 1-30] +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
    shape_observation_space = n_features * stock_dim + stock_dim + 1  # +1 for cash

    logging.info("number of validation trading dates:", len(unique_trade_dates_validation))
    logging.info("shape observation space: ", shape_observation_space)
    logging.info(f"number of columns (all features used): {n_features}, number of stocks: {stock_dim}")
    logging.info(f"unique_trade_dates_validation[0] = {unique_trade_dates_validation[0]}")

    run_count = 0
    for seed in seeds_list:
        run_count += 1
        settings.SEED_PPO = seed
        logging.info("###########################################################")
        logging.info(f"### RUN {str(run_count)} --- AGENT SEED: {str(settings.SEED_PPO)}.###")
        logging.info("###########################################################")

        #### SETUP
        ####------------------------------
        # call function create_dirs() specified above to create the directory for saving results and the trained model
        # based on the paths/parameters specified in the config.py file
        results_subdir, trained_subdir = create_dirs(mode="seed_dir", results_dir=results_dir, trained_dir=trained_dir)

        #### RUN MODEL SETUP
        ####-------------------------------
        # call run_model function to run the actual DRL algorithm
        run_model_pipeline(data=data,
                           results_dir=results_subdir,
                           trained_dir=trained_subdir,
                           stock_dim=stock_dim,
                           n_features=n_features,
                           shape_observation_space=shape_observation_space,
                           unique_trade_dates_validation=unique_trade_dates_validation)