import time
import logging
# RL models from stable-baselines
from stable_baselines3 import PPO # PPO2
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
#from stable_baselines3.ddpg.policies import DDPGPolicy
#from stable_baselines3.ppo.policies import MlpPolicy #, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise#, AdaptiveParamNoiseSpec
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.gail import generate_expert_traj, ExpertDataset  # TODO: check if they did this somewhere (?)
from config.config import paths, crisis_settings, settings, env_params, agent_params, dataprep_settings
from preprocessing.preprocessors import *
# customized env
# from env.EnvMultipleStock_train import StockEnvTrain
from env.FinancialMarketEnv import FinancialMarketEnv  # StockEnvValidation
# from env.EnvMultipleStock_trade import StockEnvTrade
import os
from model.models_ import *

#######################################
##    FUNCTION TO RUN WHOLE SETUP    ##
#######################################

# todo: combine with run_DRL and rename into run_DRL
def run_model(df,
               results_dir,
               trained_dir,
               # asset_name_column=dataprep_settings.ASSET_NAME_COLUMN,
               # date_column=dataprep_settings.DATE_COLUMN,
               stock_dim,
               n_features,
               shape_observation_space,
               unique_trade_dates_validation,
               run_mode=settings.STRATEGY_MODE,
               retrain_train_data=True,
                            # False, if we want to use the pre-trained model instead of retraining on the whole training data for each episode
               ) -> None:

    logging.warning("=================Starting {}-only Strategy================".format(settings.STRATEGY_MODE))
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_trade_state = []  # Generated with DRL_trading() at the bottom of this function
    sharpe_list = []  # list of Sharpe Ratios for the ppo agent model

    if crisis_settings.CRISIS_MEASURE is not None:
        insample_data_crisis_threshold, insample_data_subset = get_crisis_threshold(df=df,
                                                    mode="insample",
                                                    crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                    date_colname=dataprep_settings.DATE_COLUMN,
                                                    crisis_measure_colname=crisis_settings.CRISIS_MEASURE,
                                                    cutoff_Xpercentile=crisis_settings.CUTOFF_XPERCENTILE,
                                                    startdate=settings.STARTDATE_TRAIN,
                                                    enddate=settings.STARTDATE_VALIDATION)
    # for timing, calculating how long it runs
    episodes_start = time.time()

    # ------------------------
    # RUN MULTIPLE EPISODES
    # the model is trained rolling:

    # if retrain_train_data == True (Default):
    # for each episode:
    #           - split data in train, validation, test (trade) data
    #           - instantiate the environments (train, validation, trade), while passing the last state of the
    #             last trading period to the trading env, so that there is no gap.
    #               - the train env works with train data, which always starts at the beginning of the time series
    #                 and ends at a certain date which is rolled forward for 1 month for each episode
    #               - the validation env works with the rolling validation data, and starts with an otherwise empty state space
    #                 (no inheritance from the training env, state space, but the trained model is used here
    #               - the trading env starts with an empty state space as well but then inherits the last state
    #                 from the previous trading periods in previous episode
    # only the last trading state is passed so that trading can be done continuously

    # if retrain_train_data == False: the train data is not retrained, but the model is used that was trained on this data
    # this model is then further trained on the new training data and so on
    # for this, the last training state is also passed to the training env.
    # ------------------------
    current_episode_number = 0
    # for every episode (each ending at i, starting at i-rebalance_window-validation_window)
    for i in range(settings.REBALANCE_WINDOW + settings.VALIDATION_WINDOW,  # from (63+63 = 128 trading days)
                   len(unique_trade_dates_validation),
                   # total number of validation trading days # todo rename everywhere, so it makes more sense
                   settings.REBALANCE_WINDOW):  # step (63 trading days) # todo: why called rebalance and not shift_window?
        logging.warning("==================================================")
        current_episode_number += 1
        logging.warning("current episode        : "+str(current_episode_number))
        logging.warning("iteration (time step)  : "+str(i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW + 1))
        # initial state is empty
        if i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW == 0:
            # rbw and vw both 63, so if i = 126, i=126-63-63=0, etc.; initial = True
            # inital state, only holds for the first episode
            initial = True
        else:
            initial = False
        logging.warning("episode ending at iteration (i) = {}, initial episode = {}".format(i, initial))

        # Tuning turbulence index based on current data
        # Turbulence lookback window is one quarter
        # TODO: understand this
        if crisis_settings.CRISIS_MEASURE is not None:
            # since insample turbulence index is for insample data,we calculate current turbulence index
            # based on validation data
            end_date_index = \
                df.index[df["datadate"] ==
                         unique_trade_dates_validation[i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW]
                         ].to_list()[-1]
            start_date_index = end_date_index - settings.VALIDATION_WINDOW * 30 + 1
            # *30 because we assume 30 days per month (?) # todo: but trading days are less !, and why +1?
            crisis_window_enddate = df["datadate"].iloc[end_date_index]
            crisis_window_startdate = df["datadate"].iloc[start_date_index]
            crisis_threshold, _ = get_crisis_threshold(df=df,
                                                       mode="newdata",
                                                       crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                       date_colname=dataprep_settings.DATE_COLUMN,
                                                       crisis_measure_colname=crisis_settings.CRISIS_MEASURE,
                                                       insample_data_turbulence_threshold=insample_data_crisis_threshold,
                                                       insample_data_subset=insample_data_subset,
                                                       startdate=crisis_window_startdate,
                                                       enddate=crisis_window_enddate)
            logging.warning(f"crisis threshold from get_crisis_threshold (mode: newdata): {crisis_threshold}.")
        else:
            crisis_threshold = 0
            logging.warning(f"crisis threshold from get_crisis_threshold (mode: newdata): {crisis_threshold}.")

        # get data variables
        train_beginning = settings.STARTDATE_TRAIN
        train_ending = unique_trade_dates_validation[i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW]
        validation_beginning = unique_trade_dates_validation[i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW]
        validation_ending = unique_trade_dates_validation[i - settings.REBALANCE_WINDOW]
        trading_beginning = unique_trade_dates_validation[i - settings.REBALANCE_WINDOW]
        trading_ending = unique_trade_dates_validation[i]

        ############## Data Setup starts ##############
        # get training data
        train_data = split_data_by_date(df=df,
                                        start=train_beginning,
                                        end=train_ending,
                                        date_column=dataprep_settings.DATE_COLUMN,
                                        asset_name_column=dataprep_settings.ASSET_NAME_COLUMN)
        # get validation data
        validation_data = split_data_by_date(df=df, start=validation_beginning,
                                             end=validation_ending,
                                             date_column=dataprep_settings.DATE_COLUMN,
                                             asset_name_column=dataprep_settings.ASSET_NAME_COLUMN)
        # get trade data (=test data)
        trade_data = split_data_by_date(df=df,
                                        start=trading_beginning,
                                        end=trading_ending,
                                        date_column=dataprep_settings.DATE_COLUMN,
                                        asset_name_column=dataprep_settings.ASSET_NAME_COLUMN)
        logging.warning("train, validation, test split on data complete.")
        ############## Data Setup ends ##############

        ############## Environment Setup starts ##############
        # initialize training environment for the current episode
        env_train = DummyVecEnv([lambda: FinancialMarketEnv(df=train_data,  # todo: check, was StockEnvTrain
                                                            features_list=dataprep_settings.FEATURES_LIST,
                                                            # day=0,
                                                            iteration=i,  # only used for logging.info
                                                            model_name=settings.STRATEGY_MODE,
                                                            # only used for logging.info
                                                            mode="train",
                                                            crisis_measure=None,  # for TRAINING, no crisis measure used
                                                            crisis_threshold=0,
                                                            hmax_normalize=env_params.HMAX_NORMALIZE,
                                                            initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                                            transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                                            reward_scaling=env_params.REWARD_SCALING,
                                                            assets_dim=stock_dim,
                                                            shape_observation_space=shape_observation_space,
                                                            initial=True,
                                                            previous_state=[],
                                                            price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                                                            results_dir=results_dir)])
        env_train.seed(settings.SEED_AGENT)
        env_train.action_space.seed(settings.SEED_AGENT)
        # todo: check seeding
        # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/

        logging.warning("created instance env_train.")

        # initialize validation environment
        env_val = DummyVecEnv([lambda: FinancialMarketEnv(df=validation_data,  # was StockEnvTrain
                                                          features_list=dataprep_settings.FEATURES_LIST,
                                                          # day=0,
                                                          iteration=i,  # only used for logging.infoing
                                                          model_name=settings.STRATEGY_MODE,
                                                          # only used for logging.infoing
                                                          mode="validation",
                                                          crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                          crisis_threshold=crisis_threshold,
                                                          hmax_normalize=env_params.HMAX_NORMALIZE,
                                                          initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                                          transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                                          reward_scaling=env_params.REWARD_SCALING,
                                                          assets_dim=stock_dim,
                                                          shape_observation_space=shape_observation_space,
                                                          initial=True,
                                                          previous_state=[],
                                                          price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                                                          results_dir=results_dir)])
        env_val.seed(settings.SEED_AGENT)
        env_val.action_space.seed(settings.SEED_AGENT)

        # reset validation environment to obtain observations from the validation environment
        obs_val = env_val.reset()  # todo: why do we reset the validation but not the training env?
        logging.warning("created instance env_val and reset the validation env.")

        # make trading environment
        env_trade = DummyVecEnv([lambda: FinancialMarketEnv(df=trade_data,  # was StockEnvTrade
                                                            # NEW
                                                            # env-parameters to be set in config
                                                            assets_dim=stock_dim,
                                                            mode="trade",
                                                            features_list=dataprep_settings.FEATURES_LIST,
                                                            hmax_normalize=env_params.HMAX_NORMALIZE,
                                                            initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                                            transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                                            reward_scaling=env_params.REWARD_SCALING,
                                                            shape_observation_space=shape_observation_space,
                                                            crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                            crisis_threshold=crisis_threshold,
                                                            initial=initial,
                                                            previous_state=last_trade_state,
                                                            model_name=settings.STRATEGY_MODE,
                                                            iteration=i,
                                                            price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                                                            results_dir=results_dir)])
        env_trade.seed(settings.SEED_AGENT)
        env_trade.action_space.seed(settings.SEED_AGENT)
        # reset environment
        obs_trade = env_trade.reset()
        logging.warning("created instance env_trade and reset env.")
        ############## Environment Setup ends ##############

        ############## Training starts ##############
        logging.warning(f"##### TRAINING")
        logging.warning(f"---{settings.STRATEGY_MODE.upper()} training from: {train_beginning} to {train_ending}, (i={i}).")
        model_agent = DRL_train(env_train=env_train,
                              agent_name=settings.STRATEGY_MODE,
                              trained_dir=trained_dir,
                              save_name=f"episodeIndex_{i}_trainBeginning_{train_beginning}_"
                                        f"train_ending_{train_ending}")
        # trained model is saved in DRL_trading, then used in validation
        ############## Training ends ##############

        ############## Validation starts ##############
        logging.warning(f"##### VALIDATION")
        logging.warning(f"---{settings.STRATEGY_MODE.upper()} Validation from: {validation_beginning} to {validation_ending}, (i={i}).")
        DRL_predict(trained_model=model_agent,
                    test_data=validation_data,
                    test_env=env_val,
                    test_obs=obs_val,
                    mode="validation",
                    iteration=i,
                    model_name=settings.STRATEGY_MODE,
                    results_dir=results_dir)
        #sharpe_agent = get_performance_metrics(model_name=settings.STRATEGY_MODE, iteration=i, results_dir=results_dir)
        #logging.warning(f"{settings.STRATEGY_MODE.upper()} Sharpe Ratio: "+str(sharpe_agent))
        #sharpe_list.append(sharpe_agent)
        ############## Validation ends ##############

        ############## Trading starts ##############
        # trade with the chosen agent
        # learn with the other agents
        # todo: what is the difference between learn model.learn and model.predict?
        logging.warning(f"##### TRADING (TESTING)")
        logging.warning(f"---{settings.STRATEGY_MODE.upper()} Trading from: {trading_beginning} to {trading_ending}, (i={i}).======")
        last_trade_state = DRL_predict(trained_model=model_agent,
                                             test_data=trade_data,
                                             test_env=env_trade,
                                             test_obs=obs_trade,
                                             mode="trade",
                                             iteration=i,
                                             model_name=settings.STRATEGY_MODE,
                                             results_dir=results_dir)
        ############## Trading ends ##############

        # todo: for the next episode, make possible that you can choose to not retrain on the whole data again + new data
        # todo: but instead pickup the latest trained model and continue training from thereon
        # note: until now, seeding does nt yield reproducible results. that means: every time we retrain on the same dataset,
        # with the same parameters, we get different results (slightly), so the current training procedure accounts (a little)
        # for that, has a sinilar effect as if multiple different seeds are used.
        # But this is not a good option:
        # todo: fix seeding nicely, else fix seeding with a "hack"
    episodes_end = time.time()
    logging.warning(f"Single {settings.STRATEGY_MODE} "
                 f"Strategy took: {str((episodes_end - episodes_start) / 60)}, minutes.\n")


# todo: combine with run_DRL and rename into run_DRL
def run_model2(df,
               results_dir,
               trained_dir,
               # asset_name_column=dataprep_settings.ASSET_NAME_COLUMN,
               # date_column=dataprep_settings.DATE_COLUMN,
               stock_dim,
               n_features,
               shape_observation_space,
               unique_trade_dates_validation, # todo: rm
               run_mode=settings.STRATEGY_MODE,
               retrain_train_data=True,
                            # False, if we want to use the pre-trained model instead of retraining on the whole training data for each episode
               ) -> None:

    logging.warning("=================Starting {}-only Strategy (run_model2)================".format(settings.STRATEGY_MODE))
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_train_state = []  # Generated with DRL_trading() at the bottom of this function
    #last_validation_state = []  # Generated with DRL_trading() at the bottom of this function
    last_trade_state = []  # Generated with DRL_trading() at the bottom of this function
    sharpe_list = []  # list of Sharpe Ratios for the ppo agent model
    last_asset_price_train = None
    last_asset_price_trade = None


    if crisis_settings.CRISIS_MEASURE is not None:
        insample_data_crisis_threshold, insample_data_subset = get_crisis_threshold(df=df,
                                                    mode="insample",
                                                    crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                    date_colname=dataprep_settings.DATE_COLUMN,
                                                    crisis_measure_colname=crisis_settings.CRISIS_MEASURE,
                                                    cutoff_Xpercentile=crisis_settings.CUTOFF_XPERCENTILE,
                                                    startdate=settings.STARTDATE_TRAIN,
                                                    enddate=settings.STARTDATE_VALIDATION)
    # for timing, calculating how long it runs
    episodes_start = time.time()

    # ------------------------
    # RUN MULTIPLE EPISODES
    # the model is trained rolling:

    # if retrain_train_data == True (Default):
    # for each episode:
    #           - split data in train, validation, test (trade) data
    #           - instantiate the environments (train, validation, trade), while passing the last state of the
    #             last trading period to the trading env, so that there is no gap.
    #               - the train env works with train data, which always starts at the beginning of the time series
    #                 and ends at a certain date which is rolled forward for 1 month for each episode
    #               - the validation env works with the rolling validation data, and starts with an otherwise empty state space
    #                 (no inheritance from the training env, state space, but the trained model is used here
    #               - the trading env starts with an empty state space as well but then inherits the last state
    #                 from the previous trading periods in previous episode
    # only the last trading state is passed so that trading can be done continuously

    # if retrain_train_data == False: the train data is not retrained, but the model is used that was trained on this data
    # this model is then further trained on the new training data and so on
    # for this, the last training state is also passed to the training env.
    # ------------------------
    current_episode_number = 1
    last_episode = False
    #
    load_trained_model = False
    trained_model_save_path = None
    #
    datadates = df[dataprep_settings.DATE_COLUMN].unique()
    enddate = datadates[-1]
    enddate_index = df[df[dataprep_settings.DATE_COLUMN] == enddate].index.to_list()[0]

    # get data variables
    # get data indizes
    #train_beginning = settings.STARTDATE_TRAIN
    train_beginning = df[df[dataprep_settings.DATE_COLUMN] == settings.STARTDATE_TRAIN].index.to_list()[0]
    #train_ending = settings.ENDDATE_TRAIN
    train_ending = df[df[dataprep_settings.DATE_COLUMN] == settings.ENDDATE_TRAIN].index.to_list()[0]
    validation_beginning = train_ending
    validation_ending = validation_beginning + settings.VALIDATION_WINDOW
    trading_beginning = validation_ending
    trading_ending = trading_beginning + settings.TRADING_WINDOW


    while trading_beginning <= enddate_index:
        logging.warning("==================================================")
        logging.warning("current episode        : "+str(current_episode_number))
        #logging.warning("iteration (time step)  : "+str(i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW + 1))
        # initial state is empty
        if current_episode_number == 1:
            # rbw and vw both 63, so if i = 126, i=126-63-63=0, etc.; initial = True
            # inital state, only holds for the first episode
            initial = True
        else:
            initial = False
        logging.warning("--------INITIALS")
        logging.warning(f"RUN_MODE {settings.RUN_MODE}")
        logging.warning(f"episode {current_episode_number}")
        logging.warning(f"train_beginning {train_beginning}")
        logging.warning(f"train_ending {train_ending}")
        logging.warning(f"validation_beginning {validation_beginning}")
        logging.warning(f"validation_ending {validation_ending}")
        logging.warning(f"trading_beginning {trading_beginning}")
        logging.warning(f"trading_ending {trading_ending}")
        logging.warning("--------")
        logging.warning(f"enddate {enddate}, enddate index {enddate_index}")
        logging.warning("--------")
        logging.warning(f"load_trained_model {load_trained_model}")
        logging.warning(f"trained_model_save_path {trained_model_save_path}")
        logging.warning("--------")
        logging.warning(f"initial episode: {initial}")
        logging.warning("--------")

        #logging.warning("episode ending at iteration (i) = {}, initial episode = {}".format(i, initial))

        # Tuning turbulence index based on current data
        # Turbulence lookback window is one quarter
        # TODO: understand this
        if crisis_settings.CRISIS_MEASURE is not None:
            # since insample turbulence index is for insample data,we calculate current turbulence index
            # based on validation data
            #end_date_index = df.index[df["datadate"] == validation_beginning].to_list()[0]
            end_date_index = train_ending
            start_date_index = end_date_index - settings.VALIDATION_WINDOW * 30 + 1
            # *30 because we assume 30 days per month (?) # todo: but trading days are less !, and why +1?
            crisis_window_enddate = df[df.index == end_date_index][dataprep_settings.DATE_COLUMN].to_list()[0]
            crisis_window_startdate = df[df.index == start_date_index][dataprep_settings.DATE_COLUMN].to_list()[0]
            #crisis_window_startdate = df[dataprep_settings.DATE_COLUMN].iloc[start_date_index]
            crisis_threshold, _ = get_crisis_threshold(df=df,
                                                       mode="newdata",
                                                       crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                       date_colname=dataprep_settings.DATE_COLUMN,
                                                       crisis_measure_colname=crisis_settings.CRISIS_MEASURE,
                                                       insample_data_turbulence_threshold=insample_data_crisis_threshold,
                                                       insample_data_subset=insample_data_subset,
                                                       startdate=crisis_window_startdate,
                                                       enddate=crisis_window_enddate)
            logging.warning(f"crisis threshold from get_crisis_threshold (mode: newdata): {crisis_threshold}.")
        else:
            crisis_threshold = 0
            logging.warning(f"crisis threshold from get_crisis_threshold (mode: newdata): {crisis_threshold}.")

        ############## Data Setup starts ##############
        # get training data
        train_data = df[(df.index >= train_beginning) & (df.index <= train_ending)]
        if train_ending not in df.index:
            break
        # get validation data
        validation_data = df[(df.index >= validation_beginning) & (df.index <= validation_ending)]
        if validation_ending not in df.index:
            break
        # get trade data (=test data)
        trade_data = df[(df.index >= trading_beginning) & (df.index <= trading_ending)]
        if trading_beginning in df.index and trading_ending not in df.index:
            trading_ending = df.index[-1]
            last_episode = True
        logging.warning("train, validation, test split on data complete.")
        ############## Data Setup ends ##############

        ############## Environment Setup starts ##############
        # initialize training environment for the current episode
        env_train = DummyVecEnv([lambda: FinancialMarketEnv(df=train_data,  # todo: check, was StockEnvTrain
                                                            features_list=dataprep_settings.FEATURES_LIST,
                                                            day=train_beginning,
                                                            iteration=current_episode_number,  # only used for logging.info
                                                            model_name=settings.STRATEGY_MODE,
                                                            # only used for logging.info
                                                            mode="train",
                                                            crisis_measure=None,  # for TRAINING, no crisis measure used
                                                            crisis_threshold=0,
                                                            hmax_normalize=env_params.HMAX_NORMALIZE,
                                                            initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                                            transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                                            reward_scaling=env_params.REWARD_SCALING,
                                                            assets_dim=stock_dim,
                                                            shape_observation_space=shape_observation_space,
                                                            initial=initial,
                                                            previous_state=last_train_state,
                                                            previous_asset_price=last_asset_price_train,
                                                            price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                                                            results_dir=results_dir,
                                                            reset_counter=0)])

        env_train.seed(settings.SEED_AGENT)
        env_train.action_space.seed(settings.SEED_AGENT)
        # todo: check seeding
        # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/

        #train_data.to_csv(f"train_data{current_episode_number}.csv") # todo: rm
        logging.warning("created instance env_train.")

        # initialize validation environment
        env_val = DummyVecEnv([lambda: FinancialMarketEnv(df=validation_data,  # was StockEnvTrain
                                                          features_list=dataprep_settings.FEATURES_LIST,
                                                          day=validation_beginning,
                                                          iteration=current_episode_number,  # only used for logging.infoing
                                                          model_name=settings.STRATEGY_MODE,
                                                          # only used for logging.info
                                                          mode="validation",
                                                          crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                          crisis_threshold=crisis_threshold,
                                                          hmax_normalize=env_params.HMAX_NORMALIZE,
                                                          initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                                          transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                                          reward_scaling=env_params.REWARD_SCALING,
                                                          assets_dim=stock_dim,
                                                          shape_observation_space=shape_observation_space,
                                                          initial=True, # for validation, we always have an "initial state"
                                                          previous_state=[],
                                                          previous_asset_price=None,
                                                          price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                                                          results_dir=results_dir,
                                                          reset_counter=0)])
        env_val.seed(settings.SEED_AGENT)
        env_val.action_space.seed(settings.SEED_AGENT)

        # reset validation environment to obtain observations from the validation environment
        obs_val = env_val.reset()  # todo: why do we reset the validation but not the training env?
        logging.warning("created instance env_val and reset the validation env.")

        # make trading environment
        env_trade = DummyVecEnv([lambda: FinancialMarketEnv(df=trade_data,  # was StockEnvTrade
                                                            # NEW
                                                            # env-parameters to be set in config
                                                            day=trading_beginning,
                                                            assets_dim=stock_dim,
                                                            mode="trade",
                                                            features_list=dataprep_settings.FEATURES_LIST,
                                                            hmax_normalize=env_params.HMAX_NORMALIZE,
                                                            initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                                            transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                                            reward_scaling=env_params.REWARD_SCALING,
                                                            shape_observation_space=shape_observation_space,
                                                            crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                            crisis_threshold=crisis_threshold,
                                                            initial=initial,
                                                            previous_state=last_trade_state,
                                                            previous_asset_price=last_asset_price_trade,
                                                            model_name=settings.STRATEGY_MODE,
                                                            iteration=current_episode_number,
                                                            price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                                                            results_dir=results_dir,
                                                            reset_counter=0)])
        last_asset_price_trade = trade_data[dataprep_settings.MAIN_PRICE_COLUMN][
            trade_data.index == trade_data.index[-1]].values.tolist()

        env_trade.seed(settings.SEED_AGENT)
        env_trade.action_space.seed(settings.SEED_AGENT)
        # reset environment
        obs_trade = env_trade.reset()
        logging.warning("created instance env_trade and reset env.")
        ############## Environment Setup ends ##############

        ############## Training starts ##############
        logging.warning(f"##### TRAINING")
        logging.warning(f"---{settings.STRATEGY_MODE.upper()} training from: {train_beginning} to {train_ending}, "
                        f"(i={current_episode_number}).")
        model_agent, last_train_state, last_asset_price_train, trained_model_save_path = \
            DRL_train(env_train=env_train,
                      agent_name=settings.STRATEGY_MODE,
                      trained_dir=trained_dir,
                      save_name=f"episodeIndex_{current_episode_number}_trainBeginning_{train_beginning}_"
                                f"train_ending_{train_ending}",
                      results_dir=results_dir,
                      iteration=current_episode_number,
                      load_trained_model=load_trained_model,
                      trained_model_save_path=trained_model_save_path)
        #if current_episode_number <= 5:
        #    print("last asset price train 3", last_asset_price_train[:3]) # given by train data
        #    print("lat 3: ", lat[:3]) # given back by render of env
        # trained model is saved in DRL_trading, then used in validation
        ############## Training ends ##############

        ############## Validation starts ##############
        logging.warning(f"##### VALIDATION")
        logging.warning(f"---{settings.STRATEGY_MODE.upper()} Validation from: {validation_beginning} "
                        f"to {validation_ending}, (i={current_episode_number}).")
        DRL_predict(trained_model=model_agent,
                    test_data=validation_data,
                    test_env=env_val,
                    test_obs=obs_val,
                    mode="validation",
                    iteration=current_episode_number,
                    model_name=settings.STRATEGY_MODE,
                    results_dir=results_dir)

        #sharpe_agent = get_performance_metrics(model_name=settings.STRATEGY_MODE, iteration=current_episode_number,
        #                                       results_dir=results_dir)
        #logging.warning(f"{settings.STRATEGY_MODE.upper()} Sharpe Ratio: "+str(sharpe_agent))
        #sharpe_list.append(sharpe_agent)
        ############## Validation ends ##############

        ############## Trading starts ##############
        # trade with the chosen agent
        # learn with the other agents
        # todo: what is the difference between learn model.learn and model.predict?
        logging.warning(f"##### TRADING (TESTING)")
        logging.warning(f"---{settings.STRATEGY_MODE.upper()} Trading from: "
                        f"{trading_beginning} to {trading_ending}, (i={current_episode_number}).======")
        last_trade_state, lattrade = DRL_predict(trained_model=model_agent,
                                             test_data=trade_data,
                                             test_env=env_trade,
                                             test_obs=obs_trade,
                                             mode="trade",
                                             iteration=current_episode_number,
                                             model_name=settings.STRATEGY_MODE,
                                             results_dir=results_dir)
        ############## Trading ends ##############
        #if current_episode_number <= 3: # this yields the same prices => that is how it should be
        #    print("last asset price trade 3", last_asset_price_trade[:3])
         #   print("lattrade 3 ", lattrade[:3])

        if last_episode:
            break

        if settings.RUN_MODE == "ext":
            # train_beginning is not updated, stays the same as given in config.py
            # hence training period gets longer and longer
            pass
        elif settings.RUN_MODE == "st":
            train_beginning = train_ending
        # update dates for next episode
        train_ending = train_ending + settings.TRAINING_WINDOW
        validation_beginning = train_ending
        validation_ending = validation_beginning + settings.VALIDATION_WINDOW
        trading_beginning = validation_ending
        trading_ending = trading_beginning + settings.TRADING_WINDOW


        # increase current episode number
        current_episode_number += 1
        #
        load_trained_model = True
        # print reset counts:
        logging.warning(f"---env resets in episode {current_episode_number}---")
        logging.warning("train: ")
        _, _, _, reset_counts, final_state_counter, _ = env_train.render()
        logging.warning(f"env resets: {reset_counts}")
        logging.warning(f"env final_state_counter: {final_state_counter}")
        logging.warning(f"train data: {train_beginning}:{train_ending}")
        logging.warning("validation: ")
        _, _, _, reset_counts, final_state_counter, _ = env_val.render()
        logging.warning(f"validation env resets: {reset_counts}")
        logging.warning(f"env final_state_counter: {final_state_counter}")
        logging.warning(f"validation data: {validation_beginning}:{validation_ending}")
        logging.warning("trade: ")
        _, _, _, reset_counts, final_state_counter, _ = env_trade.render()
        logging.warning(f"trade env resets: {reset_counts}")
        logging.warning(f"env final_state_counter: {final_state_counter}")
        logging.warning(f"trade data: {trading_beginning}:{trading_ending}\n")




    episodes_end = time.time()
    logging.warning(f"Single {settings.STRATEGY_MODE} "
                 f"Strategy took: {str((episodes_end - episodes_start) / 60)}, minutes.\n")