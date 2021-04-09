# common library
import time

# RL models from stable-baselines
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.gail import generate_expert_traj, ExpertDataset # TODO: check if they did this somewhere (?)

from config.config import paths, crisis_settings, settings, env_params, ppo_params, dataprep_settings
from preprocessing.preprocessors import *


# customized env
#from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
#from env.EnvMultipleStock_trade import StockEnvTrade

######################################################################
# DEFINE AGENTS FOR TRAINING                                         #
######################################################################

def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{paths.TRAINED_MODELS_PATH}/{model_name}") #todo
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{paths.TRAINED_MODELS_PATH}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, save_name, trained_dir, timesteps=50000):
    """BCAP:

    PPO agent;

    Input:
    ------
    env_train   :   the (here customized) environment is passed to the agenT
    model_name  :   ?
    timesteps   :   ?

    """
    start = time.time()
    # BCAP: Policy object that implements actor critic, using a MLP (2 layers of 64), see documentation
    # Mlp = Multy-Layer-Perceptron
    # see stable-baselines doc; common
    # TODO: understand all params, also those set by default
    model = PPO2('MlpPolicy',
                 env_train,
                 ent_coef=0.005,
                 nminibatches=8,
                 seed=settings.SEED_PPO,
                 n_cpu_tf_sess=1)
    #model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{trained_dir}/{save_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

#######################################
##    FUNCTION TO RUN WHOLE SETUP    ##
#######################################
# TODO: (changed from DL_prediction to DRL_trading)
def DRL_trading(df,
                model,
                mode,
                last_state,
                iter_num,
                unique_trade_date,
                crisis_threshold,
                model_name,
                initial,
                results_dir,
                asset_dim,
                shape_observation_space):

    ######### TRADING ENVIRONMENT SETUP START ############
    # assign trading data (=test data)
    trade_data = split_data_by_date(df=df,
                                    start=unique_trade_date[iter_num - settings.REBALANCE_WINDOW],
                                    end=unique_trade_date[iter_num],
                                    date_column=dataprep_settings.DATE_COLUMN,
                                    asset_name_column=dataprep_settings.ASSET_NAME_COLUMN)
    # make trading environment
    env_trade = DummyVecEnv([lambda: StockEnvValidation(df=trade_data, # was StockEnvTrade
                                                           # NEW
                                                           # env-parameters to be set in config
                                                           assets_dim=asset_dim,
                                                           mode="trade",
                                                           features_list=dataprep_settings.FEATURES_LIST,
                                                           hmax_normalize=env_params.HMAX_NORMALIZE,
                                                           initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                                                           transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                                                           reward_scaling=env_params.REWARD_SCALING,
                                                           shape_observation_space=shape_observation_space,
                                                           crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                           crisis_threshold=crisis_threshold,
                                                           # OLD
                                                           #turbulence_threshold=turbulence_threshold,
                                                           initial=initial,
                                                           previous_state=last_state,
                                                           model_name=model_name,
                                                           iteration=iter_num,
                                                           price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                                                           results_dir=results_dir)])
    # reset environment
    obs_trade = env_trade.reset()

    ######### ENVIRONMENT SETUP START ############

    for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(obs_trade)
            #learnprint(action)
            obs_trade, rewards, dones, info = env_trade.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('{}/last_state_{}_{}_{}_i{}.csv'.format(results_dir,
                                                             mode,
                                                             model_name,
                                                             iter_num,
                                                             i), index=False)
    return last_state


def DRL_validation(model, test_data, test_env, test_obs) -> None:
    ###validation process###
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_performance_metrics(model_name,
                            iteration,
                            results_dir):
    # RENAME IN get_performance_metrics and add more metrics # todo: was get_validation_sharpe
    ###Calculate Sharpe ratio based on validation results which were saved (see validation env) ###
    df_total_value = pd.read_csv('{}/portfolio_value/end_portfolio_value_nextDayOpening_train_{}_i{}.csv'.format(
        results_dir, model_name, iteration), index_col=0)
    df_total_value.columns = ['portfolio_value']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe


#######################################
##    FUNCTION TO RUN WHOLE SETUP    ##
#######################################


# todo: combine with run_DRL and rename into run_DRL
def run_single_agent(df,
                     results_dir,
                     trained_dir,
                     asset_name_column=dataprep_settings.ASSET_NAME_COLUMN,
                     date_column=dataprep_settings.DATE_COLUMN,
                     ) -> None:


    # SETUP
    # -------
    # get parameters about dataframe shape
    stock_dim, n_features, unique_trade_dates_validation = \
        get_data_params(final_df=df,
                        asset_name_column=dataprep_settings.ASSET_NAME_COLUMN,
                        feature_cols=dataprep_settings.ALL_USED_COLUMNS,
                        date_column=dataprep_settings.DATE_COLUMN,
                        base_cols=dataprep_settings.BASE_DF_COLS,
                        startdate_validation=settings.STARTDATE_VALIDATION,
                        enddate_validation=settings.ENDDATE_VALIDATION,
                        )
    # Shape = [Current Balance]+[prices 1-30]+[owned shares 1-30] +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
    shape_observation_space = n_features * stock_dim + stock_dim + 1 # +1 for cash
    print("number of validation trading dates:", len(unique_trade_dates_validation))
    print("shape observation space: ", shape_observation_space)
    print(f"number of features: {n_features}, number of stocks: {stock_dim}")
    print(f"unique_trade_dates_validation[0] = {unique_trade_dates_validation[0]}")

    print("\n============Starting {}-only Strategy============".format(settings.STRATEGY_MODE))
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_model = [] # Generated with DRL_trading() at the bottom of this function
    sharpe_list = [] # list of Sharpe Ratios for the ppo agent model

    if crisis_settings.CRISIS_MEASURE is not None:
        insample_data_crisis_threshold, insample_data_subset = get_crisis_threshold(df=df,
                                                                mode="insample",
                                                                crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                                date_colname=dataprep_settings.DATE_COLUMN,
                                                                crisis_measure_colname=crisis_settings.CRISIS_MEASURE,
                                                                cutoff_Xpercentile=crisis_settings.CUTOFF_XPERCENTILE,
                                                                startdate=settings.STARTDATE_TRAIN,
                                                                enddate=settings.STARTDATE_VALIDATION,
                                                                )
    start = time.time()
    # for timing, calculating how long it runs

    # RUN MULTIPLE EPISODES
    # -----------------
    current_episode_number = 0
    # for every episode (each ending at i, starting at i-rebalance_window-validation_window)
    #
    for i in range(settings.REBALANCE_WINDOW + settings.VALIDATION_WINDOW,  # from (63+63 = 128 trading days)
                   len(unique_trade_dates_validation), # total number of validation trading days # todo rename everywhere, so it makes more sense
                   settings.REBALANCE_WINDOW): # step (63 trading days) # todo: why called rebalance and not shift_window?
        print("============================================")
        current_episode_number += 1
        print("current episode        : ", current_episode_number)
        print("iteration (time step)  : ", i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW + 1)
        # initial state is empty
        if i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW == 0:
            # rbw and vw both 63, so if i = 126, i=126-63-63=0, etc.; initial = True
            # inital state, only holds for the first episode
            initial = True
        else:
            initial = False
        print("Episode ending at iteration (i) = {}, initial episode = {}".format(i, initial))


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
            #enddate = unique_trade_dates_validation[i - rebalance_window - validation_window]
            # TODO: don't actually need to calculate an index, since we only want the date
            #startdate = df["datadate"].iloc[start_date_index]
            crisis_threshold, _ = get_crisis_threshold(df=df, # todo: rename to crisis_threshold
                                                    mode="newdata",
                                                    crisis_measure=crisis_settings.CRISIS_MEASURE,
                                                    date_colname=dataprep_settings.DATE_COLUMN,
                                                    crisis_measure_colname=crisis_settings.CRISIS_MEASURE,
                                                    insample_data_turbulence_threshold=insample_data_crisis_threshold,
                                                    insample_data_subset=insample_data_subset,
                                                    startdate=crisis_window_startdate,
                                                    enddate=crisis_window_enddate,
                                                    )
            print(f"\ncrisis threshold from get_crisis_threshold (mode: newdata): {crisis_threshold}.")

        ############## Environment Setup starts ##############
        # get training data
        train_data = split_data_by_date(df=df,
                                        start=settings.STARTDATE_TRAIN,
                                        end=unique_trade_dates_validation[i -
                                                                          settings.REBALANCE_WINDOW -
                                                                          settings.VALIDATION_WINDOW],
                                        date_column=dataprep_settings.DATE_COLUMN,
                                        asset_name_column=dataprep_settings.ASSET_NAME_COLUMN)
        # initialize training environment for the current episode
        env_train = DummyVecEnv([lambda: StockEnvValidation(df=train_data, # todo: check, was StockEnvTrain
                                                            features_list=dataprep_settings.FEATURES_LIST,
                                                            #day=0,
                                                            iteration=i, # only used for printing
                                                            model_name=settings.STRATEGY_MODE, # only used for printing
                                                            mode="train",
                                                            crisis_measure=None, # for TRAINING, no crisis measure used
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
        print("\ncreated instance env_train.")
        # get validation data
        validation_data = split_data_by_date(df=df, start=unique_trade_dates_validation[i - settings.REBALANCE_WINDOW -
                                                                                     settings.VALIDATION_WINDOW],
                                             end=unique_trade_dates_validation[i - settings.REBALANCE_WINDOW],
                                             date_column=dataprep_settings.DATE_COLUMN,
                                             asset_name_column=dataprep_settings.ASSET_NAME_COLUMN)
        # initialize validation environment
        env_val = DummyVecEnv([lambda: StockEnvValidation(df=validation_data, # was StockEnvTrain
                                                            features_list=dataprep_settings.FEATURES_LIST,
                                                            #day=0,
                                                            iteration=i, # only used for printing
                                                            model_name=settings.STRATEGY_MODE, # only used for printing
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
        # reset validation environment to obtain observations from the validation environment
        obs_val = env_val.reset() # todo: why do we reset the validation but not the training env?
        print("created instance env_val and reset the val. env.")
        ############## Environment Setup ends ##############

        train_beginning = settings.STARTDATE_TRAIN
        train_ending = unique_trade_dates_validation[i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW]
        validation_beginning = unique_trade_dates_validation[i - settings.REBALANCE_WINDOW - settings.VALIDATION_WINDOW]
        validation_ending = unique_trade_dates_validation[i - settings.REBALANCE_WINDOW]
        trading_beginning = unique_trade_dates_validation[i - settings.REBALANCE_WINDOW]
        trading_ending = unique_trade_dates_validation[i]

        ############## Training and Validation starts ##############

        print(f"======Model training from: {train_beginning} to {train_ending}, (i={i}).")

        print("======PPO Training========")
        model_ppo = train_PPO(env_train,
                              save_name=f"{settings.STRATEGY_MODE}_timesteps_{ppo_params.TRAINING_TIMESTEPS}_"
                                         f"episodeIndex_{i}_trainBeginning_{train_beginning}_"
                                         f"train_ending_{train_ending}",
                              timesteps=ppo_params.TRAINING_TIMESTEPS,
                              trained_dir=trained_dir)

        print(f"======PPO Validation from: {validation_beginning} to {validation_ending}, (i={i}).")
        DRL_validation(model=model_ppo,
                       test_data=validation_data,
                       test_env=env_val,
                       test_obs=obs_val)
        sharpe_ppo = get_performance_metrics(model_name=settings.STRATEGY_MODE, iteration=i, results_dir=results_dir)
        print("PPO Sharpe Ratio: ", sharpe_ppo)
        sharpe_list.append(sharpe_ppo)
        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print(f"======Trading from: {trading_beginning} to {trading_ending}, (i={i}).")
        last_state_model = DRL_trading(df=df,
                                       model=model_ppo,
                                       mode="trade",
                                       model_name=settings.STRATEGY_MODE,

                                       last_state=last_state_model,
                                       initial=initial,

                                       iter_num=i,
                                       unique_trade_date=unique_trade_dates_validation,
                                       crisis_threshold=crisis_threshold,
                                       results_dir=results_dir,
                                       asset_dim=stock_dim,
                                       shape_observation_space=shape_observation_space)
        print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Single {} Strategy took: ".format(settings.STRATEGY_MODE), (end - start) / 60, " minutes")



