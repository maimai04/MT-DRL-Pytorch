import logging
import time
import pandas as pd
import os
import numpy as np
import random
import torch

# import own libraries
from pipeline.performance_analysis_functions import calculate_performance_measures
from pipeline.support_functions import get_model
from environment.FinancialMarketEnv import FinancialMarketEnv

#######################################
##    FUNCTION TO RUN WHOLE SETUP    ##
#######################################

def run_rolling_window_setup(df: pd.DataFrame,
                             # where the results for the current run(with the current seed) are going to be saved
                             results_dir: str,
                             trained_dir: str,
                             # number of stocks used
                             assets_dim: int,
                             # shape of the observation space in the environment, created in run.py
                             shape_observation_space: int,
                             # shape of the lstm observation space in the environment, created in run.py
                             shape_lstm_observation_space: int,
                             # passing the logger instance, created in run.py
                             logger: logging,
                             # current seed
                             seed: int,

                             # PPO hyperparameters
                             net_arch: str,
                             optimizer,
                             optimizer_learning_rate,
                             max_gradient_norm,
                             total_episodes_to_train_base: int,
                             num_epochs,
                             batch_size,
                             gamma: float,
                             gae_lam: float,
                             clip: float,
                             critic_loss_coef: float,
                             entropy_loss_coef: float,

                             # environment parameters
                             env_step_version: str,
                             rebalance_penalty: float,
                             hmax_normalize: int,
                             initial_cash_balance: int,
                             transaction_fee: float,
                             reward_scaling: float,
                             reward_measure: str,

                             # dates and windows
                             global_enddate_backtesting_bull: int,
                             global_enddate_backtesting_bear: int,
                             global_startdate_train: int,
                             global_enddate_train: int,
                             validation_window: int=63,
                             testing_window: int=63,
                             roll_window: int=63,

                             # if True, we do expanding window approach, if False, we do rolling window approach (default)
                             retrain_data: bool = False,

                             strategy_mode: str = "ppoCustomBase",
                             date_column: str="datadate",
                             features_list: list = [],
                             single_features_list: list = [],
                             lstm_features_list: list=[],
                             price_column_name: str="adjcp",
                             predict_deterministic: bool=False
                             ) -> None:

    """
    This function implements an expanding window cross validation for time series.
    It works as follows:

    [train             ][test]
         [train              ][test]
              [train               ][test]
                    [train              ][test]
    ... the train window is expanded by roll_window (set at 63 days = 3 trading months, sinde one month has ~21 trading days)

    At the end of each training period, the trained model is saved. The model is then not (re-)trained on the whole train set + the additional train data,
    but rather what we use here is transfer learning => the model (which was trained on train data up to t) is loaded and ther trained only
    on the additional train set (from t to t+roll_window).
    The number of total train steps and the batch size are adjusted accordingly (since the additional training window in this setup
    is always 63 days, it does not make sense to train on this set as many times as on the first much bigger train set.

    # ------------------------
    # RUN MULTIPLE EPISODES
    # the model is trained rolling:

    # if retrain_train_data == True (Default):
    # for each episode:
    #           - split data in train, validation, test (trade) data
    #           - instantiate the environments (train, validation, test), while passing the last state of the
    #             last testing period to the testing env, so that there is no gap.
    #               - the train env works with train data, which always starts at the beginning of the time series
    #                 and ends at a certain date which is rolled forward for 1 month for each episode
    #               - the validation env works with the rolling validation data, and starts with an otherwise empty state space
    #                 (no inheritance from the training env, state space, but the trained model is used here
    #               - the testing env starts with an empty state space as well but then inherits the last state
    #                 from the previous testing periods in previous episode
    # only the last testing state is passed so that testing can be done continuously
    # ------------------------
    """
    logger.info("=======Starting {} Agent (run_model)=======".format(strategy_mode))

    #############################################################
    #                          SETUP                            #
    #############################################################
    logger.info("--Setup start..")
    ### create empty lists to be filled later
    # last train state is the last state we were in during the last training period
    # we need this state
    last_test_state = []  # Generated with DRL_trading() at the bottom of this function
    last_asset_price_test = None

    # ------------------------
    # set number of first episode to be trained; one episode is defined as one run through the training data
    # until the final state (final day) is reached. After training on this one episode is finished,
    # there is always a testing period on a test data set to get out-of-sample results
    current_episode_number = 1
    # in the beginning, we set this to False, since we are in the frst (not in the last) episode on the run for this seed
    last_episode = False
    # In the beginning, we have no trained model yet, so we cannot load it (hence set to False)
    # if our settings.RETRAIN_DATA = True, this will be set to true after the first episode and we will only continue training on
    # the additional train data; else, it will remain false and we will always retrain on the whole training batch
    load_trained_model = False
    trained_model_save_path = None

    # our data set is in "long" format:
    # index  datadate  tic   adjcp ...
    # 0      20000101   AAPL   ...
    # 0      20000101   AXP    ...
    # etc.,
    # so if we only take the date columns, we will have duplicates (actually each date number_assets times),
    # so we want to get the end time index of the data set like this:
    # get unique datadates
    datadates = df[date_column].unique()
    # get the enddate
    enddate = datadates[-1]
    # get the index of the enddate
    enddate_index = df[df[date_column] == enddate].index.to_list()[0]

    # GET TRAIN BEGINNING INDEX for the first episode
    i = 0
    while True:
        try:
            # we might have put the settings.STARTDATE_TRAIN pon a date like 20090101, which is not a trading day and hence
            # doesn't exist. So we try if we can access the index of this date; if the index list is empty, it's because
            # it is not a trading day and it will run the exception, where we are then going to try to get the index of the next date, until
            # we have found a training start that works (this only works if we are not too far off, since the STARTDATE_TRAIN is an integer,
            # and if we add 20090101 + 60 we get 20090101,=> not a date that actually exists
            train_beginning = df[df[date_column] == global_startdate_train + i].index.to_list()[0]
            break
        except:
            i += 1
            if i >= 31:
                break
    del i

    # GET TRAIN ENDING INDEX for the first episode
    i = 0
    while True:
        try:
            # note: global_enddate_train is just the enddate of the first train set before rolling forward the window
            train_ending = df[df[date_column] == global_enddate_train + i].index.to_list()[0]
            break
        except:
            i += 1
            if i >= 31:
                break
    del i
    # GET VALIDATION BEGINNING AND ENDING
    # NOTE: the validation is done within the training agent in order to enable early stopping or in general, analysis
    # after every epoch, the agent is evaluated on the validation set based on the mean rewards
    validation_beginning = train_ending
    validation_ending = validation_beginning + validation_window # 63 days like rolling window

    # GET TEST BEGINNING AND ENDING for the first episode
    testing_beginning = train_ending
    testing_ending = testing_beginning + testing_window  # testing window 63 days like rolling window

    # GET BACKTESTING BULL MARKET BEGINNING AND ENDING
    backtesting_bull_beginning = df.loc[0].index[0] # first date of dataframe, since we loaded the df with backtesting bull market data beginning as startdate
    i = 0
    while True:
        try:
            backtesting_bull_ending = df[df[date_column] == global_enddate_backtesting_bull + i].index.to_list()[0]
            break
        except:
            i += 1
            if i >= 3:
                break
    del i

    # GET BACKTESTING BEAR MARKET BEGINNING AND ENDING
    backtesting_bear_beginning = backtesting_bull_ending
    i = 0
    while True:
        try:
            backtesting_bear_ending = df[df[date_column] == global_enddate_backtesting_bear + i].index.to_list()[0]
            break
        except:
            i += 1
            if i >= 31:
                break
    del i
    logger.info("--Setup end.")


    #############################################################
    #       TRAIN / TEST LOOP FOR ONE SEED                      #
    #############################################################
    # for timing, calculating how long it runs
    episodes_start = time.time()
    logger.info(f"--Train/Val/Test loop starting @{episodes_start}.")

    # as long as the start of the testing period does not go beyond the end of our data set,
    # we do an expanding window train/test (cross validation basically)
    # (unless some other breaking conditions fulfilled, below)

    while testing_beginning <= enddate_index:
        logger.info("==================================================")
        if current_episode_number == 1:
            # if we are in the initial episode, we set "initial" = True,
            # initial mains we are in the initial episode and start from the initial state
            # this is important for the test set, because after the first episode, we don't want to start from
            # the initial state but continue from where we started
            initial = True
        else:
            # used for the test set, which always receives the state of the previous episode as the ne starting
            # state for the next episode
            initial = False

        logger.info("--------INITIALS")
        logger.info(f"CURRENT EPISODE {current_episode_number}")
        logger.info(f"train_beginning {train_beginning}")
        logger.info(f"train_ending {train_ending}")
        logger.info(f"validation_beginning {validation_beginning}")
        logger.info(f"validation_ending {validation_ending}")
        logger.info(f"testing_beginning {testing_beginning}")
        logger.info(f"testing_ending {testing_ending}")
        logger.info(f"backtesting_bull_beginning {backtesting_bull_beginning}")
        logger.info(f"backtesting_bull_ending {backtesting_bull_ending}")
        logger.info(f"backtesting_bear_beginning {backtesting_bear_beginning}")
        logger.info(f"backtesting_bear_ending {backtesting_bear_ending}")
        logger.info(f"global enddate of dataset: {enddate}, index of global enddate: {enddate_index}")
        logger.info(f"load_trained_model {load_trained_model}")
        logger.info(f"initial episode: {initial}")
        logger.info("--------")

        # set some seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        ############## Data Setup starts ##############
        logger.info(f"--Data setup starting.")
        # get training data
        train_data = df[(df.index >= train_beginning) & (df.index <= train_ending)]
        # if the train data index we want is not in the index, that means we have "run out of data"
        # and we don't have enough data to train and then test, so we stop the while loop
        if train_ending not in df.index:
            break
        # get validation data
        validation_data = df[(df.index >= validation_beginning) & (df.index <= validation_ending)]
        if validation_ending not in df.index:
            break
        # get test data (=test data)
        test_data = df[(df.index >= testing_beginning) & (df.index <= testing_ending)]
        # if only part of the test data is available we take this
        if testing_beginning in df.index and testing_ending not in df.index:
            testing_ending = df.index[-1]
            last_episode = True

        total_episodes_to_train = total_episodes_to_train_base

        logger.info("train data: ")
        logger.info(train_data.head(3))
        logger.info("val data: ")
        logger.info(validation_data.head(3))
        logger.info("test data: ")
        logger.info(test_data.head(3))
        logger.info(f"--Data setup ending.")

        ############## Data Setup ends ##############


        ############## Environment Setup starts ##############
        # initialize training environment for the current episode
        logger.info(f"--Create instance train env.")
        # NOTE: we get the environment using the function "get_environment", which
        # is imported from support_functions.py
        env_train = FinancialMarketEnv(df=df,
                                       features_list=features_list,
                                       single_features_list=single_features_list,
                                       lstm_features_list=lstm_features_list,
                                       day=train_beginning,
                                       iteration=current_episode_number,
                                       # only used for logger.info
                                       model_name=strategy_mode,
                                       mode="train",
                                       hmax_normalize=hmax_normalize,
                                       initial_cash_balance=initial_cash_balance,
                                       transaction_fee_percent=transaction_fee,
                                       reward_scaling=reward_scaling,
                                       assets_dim=assets_dim,
                                       shape_observation_space=shape_observation_space,
                                       shape_lstm_observation_space=shape_lstm_observation_space,
                                       initial=True,
                                       previous_state=[],
                                       previous_asset_price=[],
                                       price_colname=price_column_name,
                                       results_dir=results_dir,
                                       reset_counter=0,
                                       logger=logger,
                                       save_results=True,
                                       performance_calculation_window=7,
                                       step_version=env_step_version,
                                       rebalance_penalty=rebalance_penalty,
                                       reward_measure=reward_measure,
                                       total_episodes_to_train=total_episodes_to_train,
                                       net_arch=net_arch,
                                       )
        # seed the env action space, which is from gym.Env, see also: https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
        #env_train.seed(seed)
        env_train.action_space.seed(seed)
        env_train.reset() # prints a logging message, obs_train for debugging ot check if correct (now nothing printed out)
        logger.info(f"--Create instance train env finished.")

        # initialize validation environment
        logger.info(f"--Create instance validation env.")
        env_val = FinancialMarketEnv(df=validation_data,
                                     features_list=features_list,
                                     single_features_list=single_features_list,
                                     lstm_features_list=lstm_features_list,
                                     day=validation_beginning,
                                     iteration=current_episode_number,
                                     # only used for logger.info
                                     model_name=strategy_mode,
                                     mode="validation",
                                     hmax_normalize=hmax_normalize,
                                     initial_cash_balance=initial_cash_balance,
                                     transaction_fee_percent=transaction_fee,
                                     reward_scaling=reward_scaling,
                                     assets_dim=assets_dim,
                                     shape_observation_space=shape_observation_space,
                                     shape_lstm_observation_space=shape_lstm_observation_space,
                                     initial=True, # for validation, we always have an "initial state"
                                     # for validation, we always have an "initial state"
                                     previous_state=[], # and we never pass the previous state
                                     previous_asset_price=[],
                                     price_colname=price_column_name,
                                     results_dir=results_dir,
                                     reset_counter=0,
                                     logger=logger,
                                     save_results=True,
                                     performance_calculation_window=7,
                                     step_version=env_step_version,
                                     rebalance_penalty=rebalance_penalty,
                                     reward_measure=reward_measure,
                                     total_episodes_to_train=total_episodes_to_train,
                                     net_arch=net_arch,
                                     )
        #env_val.seed(seed)
        env_val.action_space.seed(seed)
        env_val.reset() # prints a logging message, obs_val for debugging ot check if correct (now nothing printed out)
        logger.info(f"--Create instance validation env finisher.")

        # make testing environment
        logger.info(f"--Create instance test env.")
        env_test = FinancialMarketEnv(df=test_data,
                                      features_list=features_list,
                                      single_features_list=single_features_list,
                                      lstm_features_list=lstm_features_list,
                                      day=testing_beginning,
                                      iteration=current_episode_number,
                                      # only used for logger.info
                                      model_name=strategy_mode,
                                      mode="test",
                                      hmax_normalize=hmax_normalize,
                                      initial_cash_balance=initial_cash_balance,
                                      transaction_fee_percent=transaction_fee,
                                      reward_scaling=reward_scaling,
                                      assets_dim=assets_dim,
                                      shape_observation_space=shape_observation_space,
                                      shape_lstm_observation_space=shape_lstm_observation_space,
                                      initial=initial,
                                      previous_state=last_test_state, # dict
                                      previous_asset_price=last_asset_price_test,
                                      price_colname=price_column_name,
                                      results_dir=results_dir,
                                      reset_counter=0,
                                      logger=logger,
                                      save_results=True,
                                      performance_calculation_window=7,
                                      step_version=env_step_version,
                                      rebalance_penalty=rebalance_penalty,
                                      reward_measure=reward_measure,
                                      total_episodes_to_train=total_episodes_to_train,
                                      net_arch=net_arch,
                                      )
        # get last asset prices from the test dataset; these are going to be used as initial state in the next episode
        # for the test set
        last_asset_price_test = test_data[price_column_name][test_data.index == test_data.index[-1]].values.tolist()

        #env_test.seed(seed)
        env_test.action_space.seed(seed)
        # reset environment to obtain first observations (state representation vector), which will be used as initial state for testing
        # later
        obs_test, lstm_obs_test = env_test.reset()
        logger.info(f"--Create instance test env finished.")
        ############## Environment Setup ends ##############


        ############## Get Model/ Algorithm ##############
        logger.info(f"--Create instance for ppo model.")
        ppo_model = get_model(train_environment=env_train,
                              validation_environment=env_val,
                              number_train_data_points=len(train_data.index.unique()),
                              shape_observation_space=shape_observation_space,
                              shape_lstm_observation_space=shape_lstm_observation_space,
                              assets_dim=assets_dim,
                              performance_save_path=os.path.join(results_dir, "training_performance"),
                              train_env_firstday=train_data.index[0],
                              val_env_firstday=validation_data.index[0],
                              load_trained_model=load_trained_model,
                              trained_model_save_path=trained_model_save_path,
                              current_episode_number=current_episode_number,
                              logger=logger,
                              gamma=gamma,
                              gae_lambda=gae_lam,
                              clip_epsilon=clip,
                              critic_loss_coef=critic_loss_coef,
                              entropy_loss_coef=entropy_loss_coef,
                              env_step_version=env_step_version,
                              net_arch=net_arch,
                              optimizer=optimizer,
                              optimizer_learning_rate=optimizer_learning_rate,
                              max_gradient_norm=max_gradient_norm,
                              total_timesteps_to_collect=len(train_data.index.unique()),
                              num_epochs=num_epochs,
                              batch_size=batch_size,
                              predict_deterministic=predict_deterministic,
                              )

        logger.info(f"RETRAIN_DATA = {retrain_data} and current_episode_number {current_episode_number}:")
        training_timesteps = (len(train_data.index.unique())-1) * total_episodes_to_train_base
        logger.info(f"\ntotal episodes to train on: {total_episodes_to_train_base}")

        logger.info(f"\ntotal timesteps to train on: {training_timesteps}")
        logger.info(f"\ntotal data length (train): {len(train_data.index.unique())}")


        logger.info(f"--Create instance for ppo model finished.")
        ############## Training starts ##############
        logger.info(f"##### TRAINING")
        logger.info(f"---{strategy_mode.upper()} training from: "
                     f"{train_beginning} to {train_ending}, "f"(i={current_episode_number}).")

        train_start = time.time()
        ppo_model.learn(total_timesteps=training_timesteps)
        train_end = time.time()
        logger.info(f"Training time ({strategy_mode.upper()}): " + str((train_end - train_start) / 60) + " minutes.")

        # save trained model
        # (if custom PPO, only the neural network is going to be saved, not any other metadata / parameters
        # from the agent; but since my custom version doesn't have any variable parameters, this doesn't matter
        model_save_name = f"{strategy_mode}_trainTimesteps_{training_timesteps}_" + \
                          f"ep{current_episode_number}_trainBeginning_{train_beginning}_trainEnding_{train_ending}"
        trained_model_save_path = f"{trained_dir}/{model_save_name}"

        # save trained model
        if strategy_mode == "ppo":
            ppo_model.save(trained_model_save_path)
        elif strategy_mode == "ppoCustomBase":
            torch.save(ppo_model.Brain.state_dict(), trained_model_save_path)
        # trained model is saved in DRL_trading, then used in validation  /testing
        ############## Training ends ##############


        ############## Testing starts ##############
        logger.info(f"##### TESTING")
        logger.info(f"---{strategy_mode.upper()} Testing from: "
                        f"{testing_beginning} to {testing_ending}, (ep={current_episode_number}).======")

        test_start = time.time()
        if net_arch == "mlplstm_separate":
            lstm_hidden_state = None
            lstm_hidden_state_actor = ppo_model.Brain.feature_extractor_actor.create_initial_lstm_state()
            lstm_hidden_state_critic = ppo_model.Brain.feature_extractor_critic.create_initial_lstm_state()
        elif net_arch == "mlplstm_shared":
            lstm_hidden_state = ppo_model.Brain.feature_extractor.create_initial_lstm_state()
            lstm_hidden_state_actor = None
            lstm_hidden_state_critic = None
        else:
            #lstm_obs = None
            lstm_hidden_state = None
            lstm_hidden_state_actor = None
            lstm_hidden_state_critic = None

        # for each time step available in the test set, we make predict actions, get the next state from the env, predict an action again etc.
        for j in range(len(test_data.index.unique())):
            # use the trained model to predict actions using the test_obs we received far above when we setup the test env
            # and run obs_test = env.reset()
            action, _, lstm_hidden_state, lstm_hidden_state_actor, lstm_hidden_state_critic = \
                ppo_model.predict(new_obs=obs_test,
                                  env_step_version=env_step_version,
                                  predict_deterministic=predict_deterministic,
                                  n_assets=assets_dim,
                                  # only applicable if we are using lstm, else None
                                  new_lstm_obs=lstm_obs_test,
                                  lstm_state = lstm_hidden_state,
                                  lstm_state_actor=lstm_hidden_state_actor,
                                  lstm_state_critic=lstm_hidden_state_critic
                                  )
            # take a step in the test environment and get the new test observation, reward, dones (a mask if terminal state True or False)
            # and info (here empty, hence _, since we don't need it)
            obs_test, lstm_obs_test, rewards, dones, _ = env_test.step(action)
            print(f"current obs_test 2: j {j}")
            print(obs_test[:5])
            if j == (len(test_data.index.unique()) - 2):
                # if we are in the pre-last state, we make env.render() which gives us the nextstate
                # and hence the last state to which we took a step
                # get the last state (flattened, then as dict, but we only need the flattened one)
                # the flattened test state is used only for saving to csv, the dict is used as initial state in the next episode for the test set
                last_test_state_flattened, _, last_test_state, _, _, _, _ = env_test.render()
                # save the last testing state as df
                df_last_state = pd.DataFrame({'last_state': last_test_state_flattened})
                df_last_state.to_csv(f'{results_dir}/last_state/last_state_test_'
                                     f'{strategy_mode}_ep{current_episode_number}.csv', index=False)
        test_end = time.time()
        logger.info(f"Testing time: " + str((test_end - test_start) / 60) + " minutes")

        ############## Testing ends ##############

        # if we have reached our last episode, we break / stop the while loop
        if last_episode:
            break
        if retrain_data == True:
            pass
            # load_trained_model will stay False, since we will retrain the model in every episode
            # on the whole train data from scratch
            # train_beginning is rolled forward by ROLL_WINDOW stays the same as given in config.py
            # hence training period gets longer and longer (expanding window)
            # and additionally, we retrain on the whole training data (don't use any transfer learning)
            train_beginning = train_beginning + roll_window
        elif retrain_data == False:
            # if we don't want to retrain, then we will use transfer learning, we will use the previously saved model
            load_trained_model = True
            # we will later load the previously trained model and continue training only on the new training data only
            # hence the new train beginning is the previous train ending
            #train_beginning = train_ending
            train_beginning = train_beginning + roll_window

        # Update dates for next episode
        # the train set is expanded by ROLL_WINDOW (63 trading days)
        train_ending = train_ending + roll_window
        validation_beginning = train_ending
        validation_ending = validation_beginning + roll_window
        testing_beginning = train_ending
        testing_ending = testing_beginning + roll_window

        # increase current episode number
        current_episode_number += 1

        # log some info
        logger.info(f"---env resets in episode {current_episode_number}---")
        logger.info("train: ")
        _, _, _, _,reset_counts, final_state_counter, _ = env_train.render()
        logger.info(f"env resets: {reset_counts}")
        logger.info(f"env final_state_counter: {final_state_counter}")
        logger.info(f"train data: {train_beginning}:{train_ending}")
        logger.info("validation: ")
        _, _, _, _,reset_counts, final_state_counter, _ = env_val.render()
        logger.info(f"validation env resets: {reset_counts}")
        logger.info(f"env final_state_counter: {final_state_counter}")
        logger.info(f"validation data: {validation_beginning}:{validation_ending}")
        logger.info("test: ")
        _, _, _, _,reset_counts, final_state_counter, _ = env_test.render()
        logger.info(f"test env resets: {reset_counts}")
        logger.info(f"env final_state_counter: {final_state_counter}")
        logger.info(f"test data: {testing_beginning}:{testing_ending}\n")

    episodes_end = time.time()
    logger.info(f"{strategy_mode} "
                 f"strategy for one seed took: {str((episodes_end - episodes_start) / 60)}, minutes.\n")

    #############################################################
    #                   BACKTESTING ON BULL MARKET              #
    #############################################################
    # first we also need to create directories

    logger.info("=================Backtesting on Bull Market================")

    os.makedirs(f"{results_dir}/backtest_bull/buy_trades")
    os.makedirs(f"{results_dir}/backtest_bull/cash_value")
    os.makedirs(f"{results_dir}/backtest_bull/datadates")
    os.makedirs(f"{results_dir}/backtest_bull/exercised_actions")
    os.makedirs(f"{results_dir}/backtest_bull/last_state")
    os.makedirs(f"{results_dir}/backtest_bull/number_asset_holdings")
    os.makedirs(f"{results_dir}/backtest_bull/policy_actions")
    os.makedirs(f"{results_dir}/backtest_bull/policy_actions_trans")
    os.makedirs(f"{results_dir}/backtest_bull/portfolio_value")
    os.makedirs(f"{results_dir}/backtest_bull/rewards")
    os.makedirs(f"{results_dir}/backtest_bull/sell_trades")
    os.makedirs(f"{results_dir}/backtest_bull/state_memory")
    os.makedirs(f"{results_dir}/backtest_bull/transaction_cost")
    os.makedirs(f"{results_dir}/backtest_bull/all_weights_cashAtEnd")
    os.makedirs(f"{results_dir}/backtest_bull/asset_equity_weights")

    # get data for backtesting
    backtesting_bull_data = df[(df.index >= backtesting_bull_beginning) & (df.index <= backtesting_bull_ending)]
    logger.info("backtest (bull) data: ")
    logger.info(backtesting_bull_data.head(3))

    # make testing environment
    env_backtesting_bull = FinancialMarketEnv(df=backtesting_bull_data,
                                              features_list=features_list,
                                              single_features_list=single_features_list,
                                              lstm_features_list=lstm_features_list,
                                              day=backtesting_bull_beginning,
                                              iteration=1,
                                              model_name=strategy_mode,
                                              mode="test",
                                              hmax_normalize=hmax_normalize,
                                              initial_cash_balance=initial_cash_balance,
                                              transaction_fee_percent=transaction_fee,
                                              reward_scaling=reward_scaling,
                                              assets_dim=assets_dim,
                                              shape_observation_space=shape_observation_space,
                                              shape_lstm_observation_space=shape_lstm_observation_space,
                                              initial=True,
                                              previous_state=[],
                                              previous_asset_price=[],
                                              price_colname=price_column_name,
                                              results_dir=os.path.join(results_dir, "backtest_bull"),
                                              reset_counter=0,
                                              logger=logger,
                                              save_results=True,
                                              performance_calculation_window=7,
                                              step_version=env_step_version,
                                              rebalance_penalty=rebalance_penalty,
                                              reward_measure=reward_measure,
                                              net_arch=net_arch,
                                              )

    env_backtesting_bull.action_space.seed(seed)
    # reset environment to obtain first observations (state representation vector)
    obs_backtest_bull, lstm_obs_backtest_bull = env_backtesting_bull.reset()

    if net_arch == "mlplstm_separate":
        lstm_hidden_state = None
        lstm_hidden_state_actor = ppo_model.Brain.feature_extractor_actor.create_initial_lstm_state()
        lstm_hidden_state_critic = ppo_model.Brain.feature_extractor_critic.create_initial_lstm_state()
    elif net_arch == "mlplstm_shared":
        lstm_hidden_state = ppo_model.Brain.feature_extractor.create_initial_lstm_state()
        lstm_hidden_state_actor = None
        lstm_hidden_state_critic = None
    else:
        lstm_hidden_state = None
        lstm_hidden_state_actor = None
        lstm_hidden_state_critic = None

    logger.info("created instance env_backtesting_bull and reset env.")
    ############## Environment Setup ends ##############

    ############## Backtesting starts ##############
    logger.info(f"##### STARTING BACKTESTING ON BULL MARKET")
    logger.info(f"---{strategy_mode.upper()} Testing from: "
                 f"{backtesting_bull_beginning} to {backtesting_bull_ending}, (ep={current_episode_number}).======")

    backtest_bull_start = time.time()
    # for each time step available in the test set, we make predict actions, get the next state from the env, predict an action again etc.
    for j in range(len(backtesting_bull_data.index.unique()) + 1):  # todo: changed from (nothing) to +1
        # use the trained model to predict actions using the test_obs we received far above when we setup the test env
        # and run obs_test = env.reset()
        # we backtest with our final trained model, the one that was trained on all train data
        action, _, lstm_hidden_state, lstm_hidden_state_actor, lstm_hidden_state_critic = \
            ppo_model.predict(new_obs=obs_backtest_bull,
                              env_step_version=env_step_version,
                              predict_deterministic=predict_deterministic,
                              n_assets=assets_dim,
                              # only if using lstm, else None
                              new_lstm_obs=lstm_obs_backtest_bull,
                              lstm_state=lstm_hidden_state,
                              lstm_state_actor=lstm_hidden_state_actor,
                              lstm_state_critic=lstm_hidden_state_critic
                              )
        # take a step in the test environment and get the new test observation, reward, dones (a mask if terminal state True or False)
        # and info (here empty, hence _, since we don't need it)
        obs_backtest_bull, lstm_obs_backtest_bull, rewards, dones, _ = env_backtesting_bull.step(action)
        if dones:
            break
    backtest_bull_end = time.time()
    logger.info(f"Backtesting time: " + str((backtest_bull_end - backtest_bull_start) / 60) + " minutes")

    #############################################################
    #                   BACKTESTING ON BEAR MARKET              #
    #############################################################
    # first we also need to create directories

    logger.info("=================Backtesting on Bear Market================")

    os.makedirs(f"{results_dir}/backtest_bear/buy_trades")
    os.makedirs(f"{results_dir}/backtest_bear/cash_value")
    os.makedirs(f"{results_dir}/backtest_bear/datadates")
    os.makedirs(f"{results_dir}/backtest_bear/exercised_actions")
    os.makedirs(f"{results_dir}/backtest_bear/last_state")
    os.makedirs(f"{results_dir}/backtest_bear/number_asset_holdings")
    os.makedirs(f"{results_dir}/backtest_bear/policy_actions")
    os.makedirs(f"{results_dir}/backtest_bear/policy_actions_trans")
    os.makedirs(f"{results_dir}/backtest_bear/portfolio_value")
    os.makedirs(f"{results_dir}/backtest_bear/rewards")
    os.makedirs(f"{results_dir}/backtest_bear/sell_trades")
    os.makedirs(f"{results_dir}/backtest_bear/state_memory")
    os.makedirs(f"{results_dir}/backtest_bear/transaction_cost")
    os.makedirs(f"{results_dir}/backtest_bear/all_weights_cashAtEnd")
    os.makedirs(f"{results_dir}/backtest_bear/asset_equity_weights")

    # get data for backtesting
    backtesting_bear_data = df[(df.index >= backtesting_bear_beginning) & (df.index <= backtesting_bear_ending)]
    logger.info("backtest (bear) data: ")
    logger.info(backtesting_bear_data.head(3))

    # make testing environment
    env_backtesting_bear = FinancialMarketEnv(df=backtesting_bear_data,
                                              features_list=features_list,
                                              single_features_list=single_features_list,
                                              lstm_features_list=lstm_features_list,
                                              day=backtesting_bear_beginning,
                                              iteration=1,
                                              model_name=strategy_mode,
                                              mode="test",
                                              hmax_normalize=hmax_normalize,
                                              initial_cash_balance=initial_cash_balance,
                                              transaction_fee_percent=transaction_fee,
                                              reward_scaling=reward_scaling,
                                              assets_dim=assets_dim,
                                              shape_observation_space=shape_observation_space,
                                              shape_lstm_observation_space=shape_lstm_observation_space,
                                              initial=True,
                                              previous_state=[],
                                              previous_asset_price=[],
                                              price_colname=price_column_name,
                                              results_dir=os.path.join(results_dir, "backtest_bear"),
                                              reset_counter=0,
                                              logger=logger,
                                              save_results=True,
                                              performance_calculation_window=7,
                                              step_version=env_step_version,
                                              rebalance_penalty=rebalance_penalty,
                                              reward_measure=reward_measure,
                                              net_arch=net_arch,
                                              )

    env_backtesting_bear.action_space.seed(seed)
    # reset environment to obtain first observations (state representation vector)
    obs_backtest_bear, lstm_obs_backtest_bear = env_backtesting_bear.reset()

    if net_arch == "mlplstm_separate":
        lstm_hidden_state = None
        lstm_hidden_state_actor = ppo_model.Brain.feature_extractor_actor.create_initial_lstm_state()
        lstm_hidden_state_critic = ppo_model.Brain.feature_extractor_critic.create_initial_lstm_state()
    elif net_arch == "mlplstm_shared":
        lstm_hidden_state = ppo_model.Brain.feature_extractor.create_initial_lstm_state()
        lstm_hidden_state_actor = None
        lstm_hidden_state_critic = None
    else:
        #lstm_obs_backtest_bear = None
        lstm_hidden_state = None
        lstm_hidden_state_actor = None
        lstm_hidden_state_critic = None
    logger.info("created instance env_backtesting_bear and reset env.")
    ############## Environment Setup ends ##############

    ############## Backtesting starts ##############
    logger.info(f"##### STARTING BACKTESTING ON BEAR MARKET")
    logger.info(f"---{strategy_mode.upper()} Testing from: "
                 f"{backtesting_bear_beginning} to {backtesting_bear_ending}, (ep={current_episode_number}).======")

    backtest_bear_start = time.time()
    # for each time step available in the test set, we make predict actions, get the next state from the env, predict an action again etc.
    for j in range(len(backtesting_bear_data.index.unique()) + 1):
        # use the trained model to predict actions using the test_obs we received far above when we setup the test env
        # and run obs_test = env.reset()
        # we backtest with our final trained model, the one that was trained on all train data
        action, _, lstm_hidden_state, lstm_hidden_state_actor, lstm_hidden_state_critic = \
            ppo_model.predict(new_obs=obs_backtest_bear,
                              env_step_version=env_step_version,
                              predict_deterministic=predict_deterministic,
                              n_assets=assets_dim,
                              # only relevant if using lstm
                              new_lstm_obs=lstm_obs_backtest_bear,
                              lstm_state=lstm_hidden_state,
                              lstm_state_actor=lstm_hidden_state_actor,
                              lstm_state_critic=lstm_hidden_state_critic
                              )
        # take a step in the test environment and get the new test observation, reward, dones (a mask if terminal state True or False)
        # and info (here empty, hence _, since we don't need it)
        obs_backtest_bear, lstm_obs_backtest_bear, rewards, dones, _ = env_backtesting_bear.step(action)
        if dones:
            break
    backtest_bear_end = time.time()
    logger.info(f"Backtesting (bear) time: " + str((backtest_bear_end - backtest_bear_start) / 60) + " minutes")


    #############################################################
    #         PERFORMANCE CALCULATION FOR CURRENT SEED          #
    #############################################################
    perf_start = time.time()
    calculate_performance_measures(run_path=results_dir,
                                   level="seed",
                                   seed=seed,
                                   mode="test",
                                   logger=logger)
    perf_end = time.time()
    logger.info(f"Performance calculation time: " + str((perf_start - perf_end) / 60) + " minutes")