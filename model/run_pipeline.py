import logging
import time
import pandas as pd
import os

import torch
from stable_baselines3.common.vec_env import DummyVecEnv

# import own libraries
from config.config import settings, data_settings, env_params, agent_params
from environment.FinancialMarketEnv import FinancialMarketEnv
from model.models import get_model

#######################################
##    FUNCTION TO RUN WHOLE SETUP    ##
#######################################

def run_expanding_window_setup(df: pd.DataFrame,
                               # where the results for the current run(with the current seed) are going to be saved
                               results_dir: str,
                               trained_dir: str,
                               # number of assets used
                               assets_dim: int,
                               # shape of the observation space in the environment
                               # (=n_stocks*n_features_per_stock+n_separate_features+n_stocks(for stock holdings/weights)+1(for cash position)
                               shape_observation_space: int,
                               # which ppo version is run: ppo (stable baselines3), ppoCustomBase, ppoCustomLSTM
                               ) -> None:
    """
    This function implements an expanding window cross validation for time series.
    It works as follows:

    [train             ][test]
    [train                   ][test]
    [train                         ][test]
    [train                               ][test]
    ... the train window is expanded by settings.ROLL_WINDOW (set at 63 days = 3 trading months, sinde one month has ~21 trading days)

    At the end of each training period, the trained model is saved. The model is then not (re-)trained on the whole train set + the additional train data,
    but rather what we use here is transfer learning => the model (which was trained on train data up to t) is loaded and ther trained only
    on the additional train set (from t to t+settings-ROLL_WINDOW).
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

    # NOTE: after each window expansion (= expansion of the training set), we can choose to
    # reset the model and run the whole model again (retrain_train_data = True) on the whole train data + new train data
    # or we can choose to save the trained model after each training period, and run the saved model only
    # on the additional training data.
    # The difference should not be big and I have ovserved that the performance is very siilar, because de facto,
    # the agent is still trained on the same data and on all of it. The only difference lies in the fact that the
    # the batch updates are now a bit in a different order (normally would go over whole data in batches, but if we
    # only train on the new subset, we have first not trained at all on this data and then train only on this additional data
    # and this could lead to the agent to "forget" older data. However, in our case this might actually be a bit beneficial,
    # because the latest dates of a time series are probably more important for what hapens next than what happened years ago.
    # Also, from a practitioners view it is interesting if this works without retraining, because if you have a very
    # large state representation (like, 250+ stocks and stock fundamendals for each of them and maybe also 100 financial indicators,
    # retraining can easily take a very long time. However it is of course not sure how the
    # algorithm would behave with such a large state representation and if data would be sufficient at all.
    # if we want to use the pre-trained model instead of retraining on the whole training data for each episode
    # ------------------------

    @param df:
    @param results_dir:
    @param trained_dir:
    @param assets_dim:
    @param shape_observation_space:
    @param run_mode:
    @param retrain_train_data:
    @return:
    """
    logging.info("=======Starting {} Agent (run_model)=======".format(settings.STRATEGY_MODE))

    #############################################################
    #                          SETUP                            #
    #############################################################
    logging.info("--Setup start..")
    ### create empty lists to be filled later
    # last train state is the last state we were in during the last training period
    # we need this state
    last_train_state = []  # Generated with DRL_trading() at the bottom of this function
    last_test_state = []  # Generated with DRL_trading() at the bottom of this function
    sharpe_list = []  # list of Sharpe Ratios for the ppo agent model
    last_asset_price_train = None
    last_asset_price_test = None

    # if retrain_train_data == False: the train data is not retrained, but the model is used that was trained on this data
    # this model is then further trained on the new training data and so on
    # for this, the last training state is also passed to the training env.
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
    datadates = df[data_settings.DATE_COLUMN].unique()
    # get the enddate
    enddate = datadates[-1]
    # get the index of the enddate
    enddate_index = df[df[data_settings.DATE_COLUMN] == enddate].index.to_list()[0]

    # GET TRAIN BEGINNING INDEX for the first episode
    i = 0
    while True:
        try:
            # we might have put the settings.STARTDATE_TRAIN pon a date like 20090101, which is not a trading day and hence
            # doesn't exist. So we try if we can access the index of this date; if the index list is empty, it's because
            # it is not a trading day and it will run the exception, where we are then going to try to get the index of the next date, until
            # we have found a training start that works (this only works if we are not too far off, since the STARTDATE_TRAIN is an integer,
            # and if we add 20090101 + 60 we get 20090101,=> not a date that actually exists
            train_beginning = df[df[data_settings.DATE_COLUMN] == settings.STARTDATE_TRAIN+i].index.to_list()[0]
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
            train_ending = df[df[data_settings.DATE_COLUMN] == settings.ENDDATE_TRAIN+i].index.to_list()[0]
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
    validation_ending = validation_beginning + settings.VALIDATION_WINDOW # 63 days like rolling window

    # GET TEST BEGINNING AND ENDING for the first episode
    testing_beginning = validation_ending
    testing_ending = testing_beginning + settings.TESTING_WINDOW  # testing window 63 days like rolling window

    # GET BACKTESTING BEGINNING AND ENDING
    backtesting_beginning = df.loc[0].index[0] # first date of dataframe, since we loaded the df with backtesting beginning as startdate
    i = 0
    while True:
        try:
            backtesting_ending = df[df[data_settings.DATE_COLUMN] == settings.ENDDATE_BACKTESTING-i].index.to_list()[0]
            # note it is -i, because we don't want to have overlappings between backtesting and training period
            break
        except:
            i += 1
            if i >= 31:
                break
    del i
    logging.info("--Setup end.")

    #############################################################
    #                    TRAIN / TEST LOOP                      #
    #############################################################
    # for timing, calculating how long it runs
    episodes_start = time.time()
    logging.info(f"--Train/Val/Test loop starting @{episodes_start}.")

    # as long as the start of the testing period does not go beyond the end of our data set,
    # we do an expanding window train/test (cross validation basically)
    # (unless some other breaking conditions fulfilled, below)
    while testing_beginning <= enddate_index:
        logging.info("==================================================")
        #logging.info("iteration (time step)  : "+str(i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW + 1))
        # initial state is empty
        if current_episode_number == 1:
            # if we are in the initial episode, we set "initial" = True, because
            # rbw and vw both 63, so if i = 126, i=126-63-63=0, etc.; initial = True
            # inital state, only holds for the first episode
            initial = True
        else:
            initial = False
        logging.info("--------INITIALS")
        logging.info(f"CURRENT EPISODE {current_episode_number}")
        logging.info(f"RUN_MODE {settings.RUN_MODE}")
        logging.info(f"train_beginning {train_beginning}")
        logging.info(f"train_ending {train_ending}")
        logging.info(f"validation_beginning {validation_beginning}")
        logging.info(f"validation_ending {validation_ending}")
        logging.info(f"testing_beginning {testing_beginning}")
        logging.info(f"testing_ending {testing_ending}")
        logging.info(f"backtesting_beginning {backtesting_beginning}")
        logging.info(f"backtesting_ending {backtesting_ending}")
        logging.info(f"global enddate of dataset: {enddate}, index of global enddate: {enddate_index}")
        logging.info(f"load_trained_model {load_trained_model}")
        logging.info(f"initial episode: {initial}")
        logging.info("--------")


        ############## Data Setup starts ##############
        logging.info(f"--Data setup starting.")
        # get training data
        train_data = df[(df.index >= train_beginning) & (df.index <= train_ending)]
        # if the train data index we want is not in the index, that means we have "run out of data"
        # and we don't have enough data to train and then test, so we stop the while loop
        if train_ending not in df.index:
            break
        print("train data: ", train_data)
        # get validation data
        validation_data = df[(df.index >= validation_beginning) & (df.index <= validation_ending)]
        if validation_ending not in df.index:
            break
        print("val data: ", validation_data)

        logging.info("train data: ", train_data.head(3))
        logging.info("val data: ", validation_data.head(3))

        # get test data (=test data)
        test_data = df[(df.index >= testing_beginning) & (df.index <= testing_ending)]
        # if only part of the test data is available we take this
        if testing_beginning in df.index and testing_ending not in df.index:
            testing_ending = df.index[-1]
            last_episode = True
        logging.info(f"--Data setup ending.")
        ############## Data Setup ends ##############

        ############## Environment Setup starts ##############
        # initialize training environment for the current episode
        logging.info(f"--Create instance train env.")
        if settings.STRATEGY_MODE == "ppo":
            env_train = DummyVecEnv([lambda: FinancialMarketEnv(df=train_data,
                                                                 features_list=data_settings.FEATURES_LIST,
                                                                 day=train_beginning,
                                                                 iteration=current_episode_number,
                                                                 # only used for logging.info
                                                                 model_name=settings.STRATEGY_MODE,
                                                                 # only used for logging.info
                                                                 mode="train",
                                                                 hmax_normalize=env_params.HMAX_NORMALIZE,
                                                                 initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                                                 transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                                                 reward_scaling=env_params.REWARD_SCALING,
                                                                 assets_dim=assets_dim,
                                                                 shape_observation_space=shape_observation_space,
                                                                 initial=initial,
                                                                 previous_state=last_train_state,
                                                                 previous_asset_price=last_asset_price_train,
                                                                 price_colname=data_settings.MAIN_PRICE_COLUMN,
                                                                 results_dir=results_dir,
                                                                 reset_counter=0)])
        elif settings.STRATEGY_MODE == "ppoCustomBase":
            env_train = FinancialMarketEnv(df=train_data,
                                            features_list=data_settings.FEATURES_LIST,
                                            day=train_beginning,
                                            iteration=current_episode_number,  # only used for logging.info
                                            model_name=settings.STRATEGY_MODE,
                                            # only used for logging.info
                                            mode="train",
                                            hmax_normalize=env_params.HMAX_NORMALIZE,
                                            initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                            transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                            reward_scaling=env_params.REWARD_SCALING,
                                            assets_dim=assets_dim,
                                            shape_observation_space=shape_observation_space,
                                            initial=initial,
                                            previous_state=last_train_state,
                                            previous_asset_price=last_asset_price_train,
                                            price_colname=data_settings.MAIN_PRICE_COLUMN,
                                            results_dir=results_dir,
                                            reset_counter=0)
        else:
            print("ERROR - no valid strategy mode passed. cannot create instance env_train.")
        env_train.seed(settings.SEED)
        env_train.action_space.seed(settings.SEED)
        # https://harald.co/2019/07/30/reproducibility-issues-using-openai-gym/
        logging.info(f"--Create instance train env finished.")

        # initialize validation environment
        logging.info(f"--Create instance validation env.")
        env_val = FinancialMarketEnv(df=validation_data,
                                      features_list=data_settings.FEATURES_LIST,
                                      day=validation_beginning,
                                      iteration=str(current_episode_number),  # only used for logging.info
                                      model_name=settings.STRATEGY_MODE,
                                      mode="validation",
                                      hmax_normalize=env_params.HMAX_NORMALIZE,
                                      initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                      transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                      reward_scaling=env_params.REWARD_SCALING,
                                      assets_dim=assets_dim,
                                      shape_observation_space=shape_observation_space,
                                      initial=True, # for validation, we always have an "initial state"
                                      previous_state=[],
                                      previous_asset_price=None,
                                      price_colname=data_settings.MAIN_PRICE_COLUMN,
                                      results_dir=results_dir,
                                      reset_counter=0)
        env_val.seed(settings.SEED)
        env_val.action_space.seed(settings.SEED)

        # reset validation environment to obtain observations from the validation environment
        obs_val = env_val.reset()
        logging.info(f"--Create instance validation env finisher.")

        # make testing environment
        logging.info(f"--Create instance test env.")
        env_test = FinancialMarketEnv(df=test_data,
                                      day=testing_beginning,
                                      assets_dim=assets_dim,
                                      mode="test",
                                      features_list=data_settings.FEATURES_LIST,
                                      hmax_normalize=env_params.HMAX_NORMALIZE,
                                      initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                      transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                      reward_scaling=env_params.REWARD_SCALING,
                                      shape_observation_space=shape_observation_space,
                                      initial=initial,
                                      # the previous test state is passed (iniital it is empty),
                                      # so we can continue testing from where we ended (e.g. with the asset holdings of that time)
                                      previous_state=last_test_state,
                                      previous_asset_price=last_asset_price_test,
                                      model_name=settings.STRATEGY_MODE,
                                      iteration=str(current_episode_number),
                                      price_colname=data_settings.MAIN_PRICE_COLUMN,
                                      results_dir=results_dir,
                                      reset_counter=0)

        # get last asset prices from the test dataset; these are going to be used as initial state in the next episode
        # for the test set
        last_asset_price_test = test_data[data_settings.MAIN_PRICE_COLUMN][test_data.index == test_data.index[-1]].values.tolist()

        env_test.seed(settings.SEED)
        env_test.action_space.seed(settings.SEED)
        # reset environment to obtain first observations (state representation vector)
        obs_test = env_test.reset()
        logging.info(f"--Create instance test env finished.")
        ############## Environment Setup ends ##############


        ############## Get Model/ Algorithm ##############
        logging.info(f"--Create instance for ppo model.")
        ppo_model = get_model(train_environment=env_train,
                              validation_environment=env_val,
                              number_train_data_points=len(train_data.index.unique()),
                              shape_observation_space=shape_observation_space,
                              assets_dim=assets_dim,
                              strategy_mode=settings.STRATEGY_MODE,
                              performance_save_path=os.path.join(results_dir, "training_performance"),
                              train_env_firstday=train_data.index[0],
                              val_env_firstday=validation_data.index[0],
                              load_trained_model=load_trained_model,
                              trained_model_save_path=trained_model_save_path)
        if not settings.RETRAIN_DATA & current_episode_number > 1:
            if settings.STRATEGY_MODE == "ppoCustomBase":
                training_timesteps = agent_params.ppoCustomBase.TOTAL_TIMESTEPS_TO_TRAIN // 4 # todo: make multiple of data set length
            elif settings.STRATEGY_MODE == "ppo":
                training_timesteps = agent_params.ppo.TRAINING_TIMESTEPS // 4 # todo: make multiple of data set length
        else:
            if settings.STRATEGY_MODE == "ppoCustomBase":
                training_timesteps = agent_params.ppoCustomBase.TOTAL_TIMESTEPS_TO_TRAIN
            elif settings.STRATEGY_MODE == "ppo":
                training_timesteps = agent_params.ppo.TRAINING_TIMESTEPS
        logging.info(f"--Create instance for ppo model finished.")


        ############## Training starts ##############
        logging.info(f"##### TRAINING")
        logging.info(f"---{settings.STRATEGY_MODE.upper()} training from: "
                     f"{train_beginning} to {train_ending}, "f"(i={current_episode_number}).")

        train_start = time.time()
        ppo_model.learn(total_timesteps=training_timesteps)
        train_end = time.time()
        logging.info(f"Training time ({settings.STRATEGY_MODE.upper()}): " + str((train_end - train_start) / 60) + " minutes.")

        # save trained model
        # (if custom PPO, only the neural network is going to be saved, not any other metadata / parameters
        # from the agent; but since my custom version doesn't have any variable parameters, this doesn't matter
        model_save_name = f"{settings.STRATEGY_MODE}_trainTimesteps_{training_timesteps}_" + \
                          f"ep{current_episode_number}_trainBeginning_{train_beginning}_trainEnding_{train_ending}"
        trained_model_save_path = f"{trained_dir}/{model_save_name}"

        # save trained model
        if settings.STRATEGY_MODE == "ppo":
            ppo_model.save(trained_model_save_path)
        elif settings.STRATEGY_MODE == "ppoCustomBase":
            torch.save(ppo_model.Brain.state_dict(), trained_model_save_path)

        # get the terminal state / observation from the environment.
        # This we will use if we don't want to retrain on the whole data set again; w
        # we can then simply get the new data and start from where we just stopped
        last_train_state_flattened, last_train_state, last_asset_price_train, _, _, _ = env_train.render()
        # Note on env: env.reset() resets to the initial state (first data point / observation) and returns that
        # env.render() returns the latest observation (and other stuff based on my customization)

        # save the last state as a dataframe
        df_last_state = pd.DataFrame({"last_state": last_train_state_flattened})
        df_last_state.to_csv(f"{results_dir}/last_state/last_state_train_{settings.STRATEGY_MODE}"
                             f"_ep{current_episode_number}.csv", index=False)
        # trained model is saved in DRL_trading, then used in validation  /testing
        ############## Training ends ##############

        ############## Validation starts ##############
        #logging.info(f"##### VALIDATION")
        #logging.info(f"---{settings.STRATEGY_MODE.upper()} Validation from: {validation_beginning} "
        #                f"to {validation_ending}, (i={current_episode_number}).")
        #DRL_predict(trained_model=model_agent,
        #            test_data=validation_data,
        #            test_env=env_val,
        #            test_obs=obs_val,
        #            mode="validation",
        #            iteration=current_episode_number,
        #            model_name=settings.STRATEGY_MODE,
        #            results_dir=results_dir)
        ############## Validation ends ##############

        ############## Testing starts ##############
        logging.info(f"##### TESTING")
        logging.info(f"---{settings.STRATEGY_MODE.upper()} Testing from: "
                        f"{testing_beginning} to {testing_ending}, (ep={current_episode_number}).======")

        test_start = time.time()
        # for each time step available in the test set, we make predict actions, get the next state from the env, predict an action again etc.
        for j in range(len(test_data.index.unique())):
            # use the trained model to predict actions using the test_obs we received far above when we setup the test env
            # and run obs_test = env.reset()
            action, _ = ppo_model.predict(obs_test)
            # take a step in the test environment and get the new test observation, reward, dones (a mask if terminal state True or False)
            # and info (here empty, hence _, since we don't need it)
            obs_test, rewards, dones, _ = env_test.step(action)
            if j == (len(test_data.index.unique()) - 2):
                # if we are in the pre-last state, we make env.render() which gives us the nextstate
                # todo:
                #  Note: at the date just before the last date, we don't take a step anymore here (since we start counting from 0)
                # and hence the last state to which we took a step

                # get the last state (flattened, then as dict, but we only need the flattened one),
                # the last asset prices and other stuff we don't need here
                last_test_state_flattened, last_test_state, _, _, _, _ = env_test.render()
                # save the last testing state as df
                df_last_state = pd.DataFrame({'last_state': last_test_state_flattened})
                df_last_state.to_csv(f'{results_dir}/last_state/last_state_test_'
                                     f'{settings.STRATEGY_MODE}_ep{current_episode_number}.csv', index=False)
        test_end = time.time()
        logging.warning(f"Testing time: " + str((test_end - test_start) / 60) + " minutes")

        #last_test_state, lattest = DRL_predict(trained_model=model_agent,
        #                                     test_data=test_data,
        #                                     test_env=env_test,
        #                                     test_obs=obs_test,
        #                                     mode="test",
        #                                     iteration=current_episode_number,
        #                                     model_name=settings.STRATEGY_MODE,
        #                                     results_dir=results_dir)

        ############## Testing ends ##############

        # if we have reached our last episode, we break / stop the while loop
        if last_episode:
            break

        if settings.RETRAIN_DATA == True:
            pass
            # load_trained_model will stay False, since we will retrain the model in every episode
            # on the whole train data from scratch
            # train_beginning is not updated, stays the same as given in config.py
            # hence training period gets longer and longer (expanding window)
            # and additionally, we retrain on the whole training data (don't use any transfer learning)
        elif settings.RETRAIN_DATA == False:
            # if we don't want to retrain, then we will use transfer learning, we will use the previously saved model
            load_trained_model = True
            # we will later load the previously trained model and continue training only on the new training data only
            # hence the new train beginning is the previous train ending
            train_beginning = train_ending

        # Update dates for next episode
        # the train set is expanded by ROLL_WINDOW (63 trading days)
        train_ending = train_ending + settings.ROLL_WINDOW
        validation_beginning = train_ending
        validation_ending = validation_beginning + settings.ROLL_WINDOW
        #trading_beginning = validation_ending
        #trading_ending = trading_beginning + settings.ROLL_WINDOW
        testing_beginning = validation_ending
        testing_ending = testing_beginning + settings.ROLL_WINDOW

        # increase current episode number
        current_episode_number += 1

        # log some info
        logging.info(f"---env resets in episode {current_episode_number}---")
        logging.info("train: ")
        _, _, _, reset_counts, final_state_counter, _ = env_train.render()
        logging.info(f"env resets: {reset_counts}")
        logging.info(f"env final_state_counter: {final_state_counter}")
        logging.info(f"train data: {train_beginning}:{train_ending}")
        logging.info("validation: ")
        _, _, _, reset_counts, final_state_counter, _ = env_val.render()
        logging.info(f"validation env resets: {reset_counts}")
        logging.info(f"env final_state_counter: {final_state_counter}")
        logging.info(f"validation data: {validation_beginning}:{validation_ending}")
        logging.info("test: ")
        _, _, _, reset_counts, final_state_counter, _ = env_test.render()
        logging.info(f"test env resets: {reset_counts}")
        logging.info(f"env final_state_counter: {final_state_counter}")
        logging.info(f"test data: {testing_beginning}:{testing_ending}\n")


    #############################################################
    #                PLOTS / PERFORMANCE CALCULATION            #
    #############################################################
    episodes_end = time.time()
    logging.info(f"{settings.STRATEGY_MODE} "
                 f"strategy for one seed took: {str((episodes_end - episodes_start) / 60)}, minutes.\n")

    #############################################################
    #                       BACKTESTING                         #
    #############################################################
    # first we also need to create directories
    os.makedirs(f"{results_dir}/backtest/buy_trades")
    os.makedirs(f"{results_dir}/backtest/cash_value")
    os.makedirs(f"{results_dir}/backtest/datadates")
    os.makedirs(f"{results_dir}/backtest/exercised_actions")
    os.makedirs(f"{results_dir}/backtest/last_state")
    os.makedirs(f"{results_dir}/backtest/number_asset_holdings")
    os.makedirs(f"{results_dir}/backtest/policy_actions")
    os.makedirs(f"{results_dir}/backtest/portfolio_weights")
    os.makedirs(f"{results_dir}/backtest/portfolio_value")
    os.makedirs(f"{results_dir}/backtest/rewards")
    os.makedirs(f"{results_dir}/backtest/sell_trades")
    os.makedirs(f"{results_dir}/backtest/state_memory")
    os.makedirs(f"{results_dir}/backtest/transaction_cost")

    logging.info("=================Starting Backtesting================")

    # get data for backtesting
    backtesting_data = df[(df.index >= backtesting_beginning) & (df.index <= backtesting_ending)]

    # make testing environment
    env_backtesting = FinancialMarketEnv(df=backtesting_data,
                                          day=backtesting_beginning,
                                          assets_dim=assets_dim,
                                          mode="test",
                                          features_list=data_settings.FEATURES_LIST,
                                          hmax_normalize=env_params.HMAX_NORMALIZE,
                                          initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                          transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                          reward_scaling=env_params.REWARD_SCALING,
                                          shape_observation_space=shape_observation_space,
                                          initial=True,
                                          previous_state=[],
                                          previous_asset_price=[],
                                          model_name=settings.STRATEGY_MODE,
                                          iteration="",
                                          price_colname=data_settings.MAIN_PRICE_COLUMN,
                                          results_dir=os.path.join(results_dir, "backtest"),
                                          reset_counter=0)

    env_backtesting.seed(settings.SEED)
    env_backtesting.action_space.seed(settings.SEED)
    # reset environment to obtain first observations (state representation vector)
    obs_backtest = env_backtesting.reset()
    logging.info("created instance env_backtesting and reset env.")
    ############## Environment Setup ends ##############

    ############## Backtesting starts ##############
    logging.info(f"##### BACKTESTING")
    logging.info(f"---{settings.STRATEGY_MODE.upper()} Testing from: "
                 f"{backtesting_beginning} to {backtesting_ending}, (ep={current_episode_number}).======")

    backtest_start = time.time()
    # for each time step available in the test set, we make predict actions, get the next state from the env, predict an action again etc.
    for j in range(len(backtesting_data.index.unique()) + 1):  # todo: changed from (nothing) to +1
        # use the trained model to predict actions using the test_obs we received far above when we setup the test env
        # and run obs_test = env.reset()
        # we backtest with our final trained model, the one that was trained on all train data
        action, _ = ppo_model.predict(obs_backtest)
        # take a step in the test environment and get the new test observation, reward, dones (a mask if terminal state True or False)
        # and info (here empty, hence _, since we don't need it)
        obs_backtest, rewards, dones, _ = env_backtesting.step(action)
        if j == (len(backtesting_data.index.unique()) - 1):  # todo: changed from -2 to -1
            # todo:
            #  Note: at the date just before the last date, we don't take a step anymore here (since we start counting from 0)
            # and hence the last state to which we took a step
            # get the last state (flattened, then as dict, but we only need the flattened one),
            # the last asset prices and other stuff we don't need here
            last_state_flattened, last_test_state, _, _, _, _ = env_backtesting.render()
            # save the last backtesting state as df
            df_last_state = pd.DataFrame({'last_state': last_state_flattened})
            df_last_state.to_csv(f'{results_dir}/last_state/last_state_test_{settings.STRATEGY_MODE}_'
                                 f'ep{current_episode_number}.csv', index=False)
    backtest_end = time.time()
    logging.warning(f"Backtesting time: " + str((backtest_end - backtest_start) / 60) + " minutes")