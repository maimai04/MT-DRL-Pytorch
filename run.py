import os
# own libraries
from pipeline.setup_functions import *
from pipeline.run_pipeline import *
from pipeline.support_functions import *
from config.config import *


############################
##     MAIN CONDITION     ##
############################

if __name__ == "__main__":

    ### set parameters:
    # Note: the parameters here are set separately, because then it is easier to overwrite them if the thing is run in colab e.g.

    # PATHS
    results_path = paths.RESULTS_PATH
    trained_models_path = paths.TRAINED_MODELS_PATH
    country = data_settings.COUNTRY
    data_path = paths.DATA_PATH
    raw_data_path = paths.RAW_DATA_PATH
    preprocessed_data_path = paths.PREPROCESSED_DATA_PATH
    trained_models_path = paths.TRAINED_MODELS_PATH
    results_path = paths.RESULTS_PATH
    subdir_names = paths.SUBSUBDIR_NAMES
    preprocessed_data_file = paths.PREPROCESSED_DATA_FILE

    # GENERAL SETTINGS
    now = settings.NOW
    seeds_list = settings.SEEDS_LIST
    features_list = data_settings.FEATURES_LIST
    single_features_list = data_settings.SINGLE_FEATURES_LIST
    lstm_features_list = data_settings.LSTM_FEATURES_LIST

    # DATA SETTINGS
    date_column_name = data_settings.DATE_COLUMN
    price_column_name = data_settings.MAIN_PRICE_COLUMN

    # STRATEGY SETTINGS
    strategy_mode = settings.STRATEGY_MODE
    reward_measure = settings.REWARD_MEASURE
    retrain_data = settings.RETRAIN_DATA
    run_mode = settings.RUN_MODE
    features_mode = data_settings.FEATURES_MODE

    # DATES
    global_startdate_train = settings.STARTDATE_TRAIN
    global_enddate_train = settings.ENDDATE_TRAIN
    #TODO
    startdate_backtesting_bull = settings.STARTDATE_BACKTESTING_BULL
    enddate_backtesting_bull = settings.ENDDATE_BACKTESTING_BULL

    startdate_backtesting_bear = settings.STARTDATE_BACKTESTING_BEAR
    enddate_backtesting_bear = settings.ENDDATE_BACKTESTING_BEAR

    roll_window = settings.ROLL_WINDOW
    validation_window = settings.VALIDATION_WINDOW
    testing_window = settings.TESTING_WINDOW

    # ENVIRONMENT PARAMETERS
    env_step_version = env_params.STEP_VERSION
    hmax_normalize = env_params.HMAX_NORMALIZE
    initial_cash_balance = env_params.INITIAL_CASH_BALANCE
    transaction_fee = env_params.TRANSACTION_FEE_PERCENT
    reward_scaling = env_params.REWARD_SCALING
    rebalance_penalty = env_params.REBALANCE_PENALTY

    # HP TUNING PARAMS 3 todo: rm
    now_hptuning = hptuning_config.now_hptuning
    only_hptuning = hptuning_config.only_hptuning
    gamma_list = hptuning_config.GAMMA_LIST
    gae_lam_list = hptuning_config.GAE_LAMBDA_LIST
    clip_list = hptuning_config.CLIP_EPSILON_LIST
    critic_loss_coef_hpt = hptuning_config.CRITIC_LOSS_COEF_LIST
    entropy_loss_coef_hpt = hptuning_config.ENTROPY_LOSS_COEF_LIST

    # AGENT PARAMS & HYPERPARAMETER
    net_arch = agent_params.ppoCustomBase.NET_ARCH
    batch_size = agent_params.ppoCustomBase.BATCH_SIZE
    num_epochs = agent_params.ppoCustomBase.NUM_EPOCHS
    optimizer = agent_params.ppoCustomBase.OPTIMIZER
    optimizer_learning_rate = agent_params.ppoCustomBase.OPTIMIZER_LEARNING_RATE
    gamma = agent_params.ppoCustomBase.GAMMA
    gae_lam = agent_params.ppoCustomBase.GAE_LAMBDA
    clip_epsilon = agent_params.ppoCustomBase.CLIP_EPSILON
    clip = clip_epsilon # todo
    critic_loss_coef = agent_params.ppoCustomBase.CRITIC_LOSS_COEF
    entropy_loss_coef = agent_params.ppoCustomBase.ENTROPY_LOSS_COEF
    max_gradient_norm = agent_params.ppoCustomBase.MAX_GRADIENT_NORMALIZATION
    predict_deterministic = agent_params.ppoCustomBase.PREDICT_DETERMINISTIC

    # LEARNING PARAMS
    total_timesteps_to_collect = agent_params.ppoCustomBase.TOTAL_TIMESTEPS_TO_COLLECT
    total_episodes_to_train_base = agent_params.ppoCustomBase.TOTAL_EPISODES_TO_TRAIN_BASE
    total_episodes_to_train_cont = agent_params.ppoCustomBase.TOTAL_EPISODES_TO_TRAIN_CNT


    #### SETUP DIRECTORIES AND LOGGINGS
    ####-------------------------------
    # create directories to save results and trained models for all seeds for this whole run
    # these directories are created here because they use the current timestamp (settings.NOW) defined in config.py
    results_dir, trained_dir = create_dirs(mode="run_dir",
                                           results_path=results_path,
                                           trained_models_path=trained_models_path,
                                           now=now,
                                           features_mode=features_mode,
                                           run_mode=run_mode,
                                           reward_measure=reward_measure,
                                           net_arch=net_arch,
                                           env_step_version=env_step_version,
                                           predict_deterministic=predict_deterministic
                                           )

    # create saving path and directory for loggings
    logsave_path = os.path.join(results_dir, "_LOGGINGS")
    os.makedirs(logsave_path)

    # write all parameters from config.py file into the config_log.txt file in folder _LOGGING
    # so later this file can be consulted to uniquely identify each run and its configurations
    config_logging_to_txt(results_subdir=results_dir,
                          trained_subdir=trained_dir,
                          logsave_path=logsave_path,
                          now=now,
                          seeds_list=seeds_list,
                          strategy_mode =strategy_mode,
                          reward_measure =reward_measure,
                          retrain_data =retrain_data,
                          run_mode = run_mode,
                          global_startdate_train = global_startdate_train,
                          global_enddate_train = global_enddate_train,
                          roll_window = roll_window,
                          startdate_backtesting_bull = startdate_backtesting_bull,
                          enddate_backtesting_bull = enddate_backtesting_bull,
                          startdate_backtesting_bear = startdate_backtesting_bear,
                          enddate_backtesting_bear = enddate_backtesting_bear,
                          env_step_version = env_step_version,
                          rebalance_penalty=rebalance_penalty,
                          hmax_normalize = hmax_normalize,
                          initial_cash_balance = initial_cash_balance,
                          transaction_fee = transaction_fee,
                          reward_scaling =reward_scaling,
                          country = country,
                          features_list = features_list,
                          single_features_list = single_features_list,
                          lstm_features_list=lstm_features_list,
                          features_mode =features_mode,
                          data_path = data_path,
                          raw_data_path = raw_data_path,
                          preprocessed_data_path = preprocessed_data_path,
                          trained_models_path = trained_models_path,
                          results_path = results_path,
                          subdir_names =subdir_names,
                          preprocessed_data_file =preprocessed_data_file,
                          now_hptuning = now_hptuning,
                          only_hptuning = only_hptuning,
                          gamma_list = gamma_list,
                          gae_lam_list = gae_lam_list,
                          clip_list = clip_list,
                          critic_loss_coef_hpt = critic_loss_coef_hpt,
                          entropy_loss_coef_hpt = entropy_loss_coef_hpt,
                          net_arch = net_arch,
                          batch_size = batch_size,
                          num_epochs = num_epochs,
                          optimizer = optimizer,
                          optimizer_lr = optimizer_learning_rate,
                          gamma = gamma,
                          gae_lam = gae_lam,
                          clip_epsilon = clip_epsilon,
                          critic_loss_coef = critic_loss_coef,
                          entropy_loss_coef = entropy_loss_coef,
                          max_gradient_norm = max_gradient_norm,
                          total_episodes_to_train_base = total_episodes_to_train_base,
                          total_episodes_to_train_cont = total_episodes_to_train_cont,
                          predict_deterministic=predict_deterministic
                        )


    #### LOAD PREPROCESSED DATA
    ####-------------------------------
    # call function to get the preprocessed data
    data = load_dataset(file_path=preprocessed_data_file,
                        col_subset=features_list + single_features_list,
                        date_subset="datadate",
                        date_subset_startdate=startdate_backtesting_bull,
                        asset_name_column="tic")
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
    data = data.sort_values(["datadate", "tic"])
    data.index = data[date_column_name].factorize()[0]

    # get parameters about dataframe shape (we need this information to feed it to the environment later)
    assets_dim, n_features, n_single_features, n_features_lstm, n_single_features_lstm = \
        get_data_params(final_df=data,
                        asset_name_column="tic",
                        feature_cols=features_list,
                        single_feature_cols=single_features_list,
                        lstm_cols=lstm_features_list,
                        #date_column=data_settings.DATE_COLUMN,
                        #base_cols=data_settings.BASE_DF_COLS,
                        )
    # Define the shape of the observation space (this we also need to provide to the environment to create the state space)
    # Shape = [Current Balance]+[prices 1-30]+[owned shares 1-30] +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
    shape_observation_space = n_features * assets_dim + assets_dim + 1 + n_single_features # +1 for cash
    shape_lstm_observation_space = n_features_lstm * assets_dim + n_single_features_lstm

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
    common_logger.info("(main) shape lstm observation space: " + str(shape_lstm_observation_space))
    common_logger.info(f"(main) number of columns (all features used): {str(n_features)}, number of stocks: {str(assets_dim)}")

    # begin the run count (one whole run per seed)
    # Note: the seeds are needed because there is some randomness involved in the whole process, namely:
    # in the beginning, our policy is basically random and samples from the defined action space.
    run_count = 0
    for seed in settings.SEEDS_LIST:
        run_count += 1

        logger = custom_logger(seed=seed,
                               logging_path=logsave_path,
                               level=logging.DEBUG,
                               )
        logger.info("#################################################################")
        logger.info(f"### (main) RUN {str(run_count)} --- AGENT SEED: {str(seed)}")
        logger.info("#################################################################")

        #### SETUP
        ####------------------------------
        # call function create_dirs() specified above to create the directory for saving results and the trained model
        # based on the paths/parameters specified in the config.py file
        results_subdir, trained_subdir = create_dirs(mode="seed_dir",
                                                     results_dir=results_dir,
                                                     trained_dir=trained_dir,
                                                     seed=seed,
                                                     subdir_names=subdir_names)

        #### RUN MODEL SETUP
        ####-------------------------------
        # run the chosen setup (here: expanding window)
        logger.info("Run expanding window setup.")
        run_rolling_window_setup(df=data,
                                   results_dir=results_subdir,
                                   trained_dir=trained_subdir,
                                   assets_dim=assets_dim,
                                   shape_observation_space=shape_observation_space,
                                   shape_lstm_observation_space=shape_lstm_observation_space,

                                   logger=logger,
                                   strategy_mode=strategy_mode,
                                   seed=seed,
                                   date_column=date_column_name,

                                   global_startdate_train=global_startdate_train,
                                   global_enddate_train=global_enddate_train,
                                   validation_window=validation_window,
                                   testing_window=testing_window,
                                   roll_window=roll_window,
                                   global_enddate_backtesting_bull=enddate_backtesting_bull,
                                   global_enddate_backtesting_bear=enddate_backtesting_bear,
                                   retrain_data=retrain_data,

                                   now_hptuning=hptuning_config.now_hptuning,
                                   only_hptuning=hptuning_config.only_hptuning,

                                   gamma=gamma,
                                   gae_lam=gae_lam,
                                   clip=clip_epsilon,
                                   critic_loss_coef=critic_loss_coef,
                                   entropy_loss_coef=entropy_loss_coef,
                                   total_episodes_to_train_base=total_episodes_to_train_base,
                                   total_episodes_to_train_cont=total_episodes_to_train_cont,

                                   net_arch=net_arch,
                                   optimizer=optimizer,
                                   optimizer_learning_rate=optimizer_learning_rate,
                                   max_gradient_norm=max_gradient_norm,
                                   price_column_name=price_column_name,

                                   num_epochs=num_epochs,
                                   batch_size=batch_size,
                                   env_step_version=env_step_version,
                                   rebalance_penalty=rebalance_penalty,
                                   hmax_normalize=hmax_normalize,
                                   initial_cash_balance=initial_cash_balance,
                                   transaction_fee=transaction_fee,
                                   reward_scaling=reward_scaling,
                                   reward_measure=reward_measure,
                                   features_list=features_list,
                                   single_features_list=single_features_list,
                                   lstm_features_list=lstm_features_list,

                                 predict_deterministic=predict_deterministic,
        )




    if not hptuning_config.only_hptuning:
        #############################################################
        #         PERFORMANCE CALCULATION ACROSS ALL SEEDS          #
        #############################################################
        common_logger.info("Summarizing whole run performance across all seeds.")
        perf_start = time.time()
        calculate_performance_measures(run_path=results_dir,
                                       level="run",
                                       seed=seed,
                                       mode="test",
                                       logger=common_logger,
                                       seeds_list=seeds_list)
        common_logger.info("Summarizing whole run performance across all seeds finished.")
        perf_end = time.time()
        common_logger.info(f"Performance summary over all seeds took: " + str((perf_start - perf_end) / 60) + " minutes")
