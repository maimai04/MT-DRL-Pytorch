import os
# own libraries
from pipeline.setup_functions import *
from pipeline.run_pipeline import *
from pipeline.support_functions import *


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

    # get parameters about dataframe shape (we need this information to feed it to the environment later)
    # todo: n_features maybe not needed
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

    # we want to be able to summarize some metrics / üerformance metrics and plot some plots
    # for all seeds together (e.g. make a plot of changes in portfolio value vs. time for each seed in order to compare,
    # and calculate medium reward, max. drawdown for all seeds, sharpe ratio
    # and because I have to run many runs, I don't want to do this separately for every run by hand
    # However, that also means that the whole run is going to take a little longer (since there will be a small time loss
    # for plotting / calculating summaries

    common_logger = custom_logger(seed="allSeeds",
                                  logging_path=logsave_path,
                                  level=logging.DEBUG)
    # loggings (these are all saved in the _LOGGINGS folder of the whole run
    common_logger.info("(main) FINAL INPUT DATAFRAME")
    common_logger.info("---------------------------------------------------------")
    common_logger.info(data.head(3))
    common_logger.info(f"(main) Shape of Dataframe (unique dates, columns) : ({len(data.index.unique())}, {len(data.columns)})")
    # logging.info("(main) number of validation trading dates: " + str(len(unique_trade_dates_validation)))
    common_logger.info("(main) shape observation space: " + str(shape_observation_space))
    common_logger.info(f"(main) number of columns (all features used): {str(n_features)}, number of stocks: {str(assets_dim)}")

    # begin the run count (one whole run per seed)
    # Note: the seeds are needed because there is some randomness involved in the whole process, namely:
    # in the beginning, our policy is basically random and samples from the defined action space.
    run_count = 0
    for seed in settings.SEEDS_LIST:
        run_count += 1
        settings.SEED = seed
        # set some seeds
        #torch.manual_seed(seed)
        #np.random.seed(seed)
        #random.seed(seed)
        #torch.cuda.manual_seed(seed)
        #torch.backends.cudnn.deterministic = True

        #logging.basicConfig(filename=os.path.join(logsave_path, f"run_log_seed_{seed}"),
        #                    filemode='a',
        #                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        #                    # asctime = time, msecs, name = name of who runned it (?), levelname (e.g. DEBUG, INFO => verbosity), message
        #                    datefmt='%H:%M:%S',
        #                    # level=logging.INFO
        #                    # level=logging.DEBUG
        #                    level=logging.NOTSET)
        logger = custom_logger(seed=seed,
                               logging_path=logsave_path,
                               level=logging.DEBUG)
        logger.info("#################################################################")
        logger.info(f"### (main) RUN {str(run_count)} --- AGENT SEED: {str(settings.SEED)}")
        logger.info("#################################################################")

        #### SETUP
        ####------------------------------
        # call function create_dirs() specified above to create the directory for saving results and the trained model
        # based on the paths/parameters specified in the config.py file
        results_subdir, trained_subdir = create_dirs(mode="seed_dir", results_dir=results_dir, trained_dir=trained_dir)

        #### RUN MODEL SETUP
        ####-------------------------------
        # run the chosen setup (here: expanding window)
        logger.info("Run expanding window setup.")
        run_expanding_window_setup(df=data,
                                   results_dir=results_subdir,
                                   trained_dir=trained_subdir,
                                   assets_dim=assets_dim,
                                   #n_features=n_features,
                                   shape_observation_space=shape_observation_space,
                                   logger=logger
                                   )

    #############################################################
    #         PERFORMANCE CALCULATION ACROSS ALL SEEDS          #
    #############################################################
    common_logger.info("Summarizing whole run performance across all seeds.")
    perf_start = time.time()
    calculate_performance_measures(run_path=results_dir,
                                   level="run",
                                   seed=settings.SEED,
                                   mode="test",
                                   logger=common_logger)
    common_logger.info("Summarizing whole run performance across all seeds finished.")
    perf_end = time.time()
    common_logger.info(f"Performance summary over all seeds took: " + str((perf_start - perf_end) / 60) + " minutes")
