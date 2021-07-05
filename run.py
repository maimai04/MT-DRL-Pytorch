import logging
from setup_functions import *
from model.run_pipeline import *
import os
import numpy as np

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

    #### SETUP DIRECTORIES AND LOGGINGS
    ####-------------------------------
    # create directories to save results and trained models for all seeds for this whole run
    # these directories are created here because they use the current timestamp (settings.NOW) defined in config.py
    results_dir, trained_dir = create_dirs(mode="run_dir")

    # create saving path and directory for loggings
    logsave_path = os.path.join(results_dir, "_LOGGINGS")
    os.makedirs(logsave_path)

    # write all parameters from config.py file into the config_log.txt file in folder _LOGGING
    # so later this file can be consulted to uniquely identify each run and its configurations
    config_logging_to_txt(results_dir, trained_dir, logsave_path=logsave_path)
    logging.basicConfig(filename=os.path.join(logsave_path, "run_log"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        # asctime = time, msecs, name = name of who runned it (?), levelname (e.g. DEBUG, INFO => verbosity), message
                        datefmt='%H:%M:%S',
                        #level=logging.INFO
                        #level=logging.DEBUG
                        level=logging.NOTSET)

    #### LOAD PREPROCESSED DATA
    ####-------------------------------
    # call function to get the preprocessed data
    data = load_dataset(file_path=paths.PREPROCESSED_DATA_FILE,
                        col_subset=data_settings.FEATURES_LIST,
                        date_subset="datadate",
                        date_subset_startdate=settings.STARTDATE_BACKTESTING)
    # sort values first based on date, then on ticker (stock short-name), then factorize index based on datadate
    # (set the data index to be 0 for the first date (e.g. 20090102), 1 for the ext date etc.)
    # the goal is that we get a data set like this (for easier accessing in the environment)
    # index  datadate  tic  adjcp  ...
    #   0    20090102  AAPL  ...
    #   0    20090102  AMGN  ...
    #   0    20090102  AXP   ...
    # ...
    #   1    20090103  AAPL  ...
    #   1    20090103  AMGN  ...
    #   1    20090103  AXP   ...
    # ...
    data = data.sort_values([data_settings.DATE_COLUMN, data_settings.ASSET_NAME_COLUMN])
    data.index = data[data_settings.DATE_COLUMN].factorize()[0]

    # loggings (these are all saved in the _LOGGINGS folder of the whole run
    logging.info("(main) FINAL INPUT DATAFRAME")
    logging.info("---------------------------------------------------------")
    logging.info(data.head(3))
    logging.info(f"(main) Shape of Dataframe (unique dates, columns) : ({len(data.index.unique())}, {len(data.columns)})")

    # get parameters about dataframe shape (we need this information to feed it to the environment later)
    assets_dim, n_features = \
        get_data_params(final_df=data,
                        asset_name_column=data_settings.ASSET_NAME_COLUMN,
                        feature_cols=data_settings.FEATURES_LIST,
                        #date_column=data_settings.DATE_COLUMN,
                        #base_cols=data_settings.BASE_DF_COLS,
                        )
    # Define the shape of the observation space (this we also need to provide to the environment to create the state space)
    # Shape = [Current Balance]+[prices 1-30]+[owned shares 1-30] +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
    shape_observation_space = n_features * assets_dim + assets_dim + 1  # +1 for cash

    #logging.info("(main) number of validation trading dates: " + str(len(unique_trade_dates_validation)))
    logging.info("(main) shape observation space: "+str(shape_observation_space))
    logging.info(f"(main) number of columns (all features used): {str(n_features)}, number of stocks: {str(assets_dim)}")
    #logging.info(f"(main) unique_trade_dates_validation[0] = {str(unique_trade_dates_validation[0])}\n ")

    # we want to be able to summarize some metrics / erformance metrics and plot some plots
    # for all seeds together (e.g. make a plot of chages in portfolio value vs. time for each seed in order to compare,
    # and calculate medium reward, max. drawdown for all seeds, sharpe ratio
    # and because I have to run many runs, I don't want to do this separately for every run by hand
    # However, that also means that the whole run is going to take a little longer (since there will be a small time loss
    # for plotting / calculating summaries


    # begin the run count (one whole run per seed)
    # Note: the seeds are needed because there is some randomness involved in the whole process, namely:
    # in the beginning, our policy is basically random and samples from the defined action space.
    run_count = 0
    for seed in settings.SEEDS_LIST:
        run_count += 1
        settings.SEED = seed

        # set some seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        logging.info("\n#################################################################")
        logging.info(f"### (main) RUN {str(run_count)} --- AGENT SEED: {str(settings.SEED)}")
        logging.info("#################################################################")

        #### SETUP
        ####------------------------------
        # call function create_dirs() specified above to create the directory for saving results and the trained model
        # based on the paths/parameters specified in the config.py file
        results_subdir, trained_subdir = create_dirs(mode="seed_dir", results_dir=results_dir, trained_dir=trained_dir)

        #### RUN MODEL SETUP
        ####-------------------------------
        # run the chosen setup (here: expanding window)
        logging.info("Run expanding window setup.")
        run_expanding_window_setup(df=data,
                                   results_dir=results_subdir,
                                   trained_dir=trained_subdir,
                                   assets_dim=assets_dim,
                                   #n_features=n_features,
                                   shape_observation_space=shape_observation_space,
                                   )