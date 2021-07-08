from pipeline.setup_functions import *
from model.models_pipeline import *

"""
run only PPO
everything else c.p.

additional changes from BCAP:
- added comments
- added variable which_run, to save subsequent runs in corresponding folders.

"""

############################
##     MAIN CONDITION     ##
############################

if __name__ == "__main__":

    #### LOAD / LOAD & PREPROCESS DATA
    ####-------------------------------
    # call function to get the raw data and preprocess it, depending on params in config.py
    data = data_handling(
        # PASSING NAMES OF OPTIONAL FUNCTIONS TO BE USED in preprocessing pipeline
        calculate_price_volume_func="calculate_price_volume_WhartonData",
        add_technical_indicator_func="add_technical_indicator_with_StockStats",
        add_other_features_func="add_other_features",
        add_ANN_features_func=None,
        add_crisis_measure_func="add_crisis_measure",
        # PASSING PARAMETERS FOR EACH OPTIONAL FUNCTION
        calculate_price_volume_func_params={"new_cols_subset": dataprep_settings.NEW_COLS_SUBSET,
                                            "target_subset": None},
        add_technical_indicator_func_params={"technical_indicators_list": dataprep_settings.TECH_INDICATORS},
        add_other_features_func_params={"feature": ["returns_volatility", "log_return_daily"],
                                        "window_days": 7},
        add_ANN_features_func_params={},
        add_crisis_measure_func_params={"crisis_measure": crisis_settings.CRISIS_MEASURE},
        # ----- LEAVE -----
        preprocess_anew=dataprep_settings.PREPROCESS_ANEW,
        preprocessed_data_file=paths.PREPROCESSED_DATA_FILE,
        save_path=paths.PREPROCESSED_DATA_PATH,
        raw_data_file=paths.RAW_DATA_FILE,
        col_subset=dataprep_settings.RAW_DF_COLS_SUBSET,
        date_subset=dataprep_settings.DATE_COLUMN,
        date_subset_startdate=settings.STARTDATE_TRAIN,
    )
