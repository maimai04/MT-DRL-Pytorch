import time
# RL models from stable-baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.policies import MlpPolicy
import gym
import math
import logging
import glob
import ffn
import pandas as pd
import numpy as np

# own libraries
from config.config import *
from environment.FinancialMarketEnv import FinancialMarketEnv
from model.CustomOnPolicyBuffer import OnPolicyBuffer
from model.CustomPPOAlgorithm import PPO_algorithm
from model.CustomActorCriticNets import BrainActorCritic, \
    init_weights_feature_extractor_net, init_weights_actor_net, init_weights_critic_net

########################################################################
# DEFINE FUNCTION FOR TRAINING                                         #
########################################################################

def get_environment(df: pd.DataFrame,
                    day: int,
                    iteration: int,  # current episode number, only used for logger.info
                    mode: str, # validation, test
                    assets_dim: int, # number of assets
                    shape_observation_space: int, # n_assets*n_features_per_asset + 1(for cash) + n_assets (for asset holdings)
                    initial: bool,  # for validation, we always have an "initial state"
                    results_dir: str,
                    reset_counter: int,
                    # env and agent parameters
                    strategy_mode: str,
                    features_list: list,
                    single_features_list: list,
                    env_step_version: str,
                    rebalance_penalty: float,
                    hmax_normalize: int,
                    initial_cash_balance: int,
                    transaction_fee: float,
                    reward_scaling: float,
                    price_column_name: str,
                    reward_measure: str,
                    logger=None,
                    previous_state: list = [],  # only applies if RETRAIN = False and/or if its the test env
                    previous_asset_price: list = [],
                    save_results: bool=True,
                    calculate_sharpe_ratio: bool=False,
                    ) -> gym.Env:

    if strategy_mode == "ppo": # todo: rm
        # ppo from stable baselines returns an error when not used with the vec environment; however, since we only pass one env,
        # it doesn't actually make a difference (for games and complex problems, ppo is often used with parallel threads, but I did not
        # implement parallel workers for my work
        env = DummyVecEnv([lambda: FinancialMarketEnv(df=df,
                                                      features_list=features_list,
                                                      day=day,
                                                      iteration=iteration,
                                                      # only used for logger.info
                                                      model_name=strategy_mode,
                                                      mode=mode,
                                                      hmax_normalize=hmax_normalize,
                                                      initial_cash_balance=initial_cash_balance,
                                                      transaction_fee_percent=transaction_fee,
                                                      reward_scaling=reward_scaling,
                                                      assets_dim=assets_dim,
                                                      shape_observation_space=shape_observation_space,
                                                      initial=initial,
                                                      # for validation, we always have an "initial state"
                                                      previous_state=previous_state,
                                                      previous_asset_price=previous_asset_price,
                                                      price_colname=price_column_name,
                                                      results_dir=results_dir,
                                                      reset_counter=reset_counter,
                                                      logger=logger,
                                                      save_results=save_results,
                                                      calculate_sharpe_ratio=calculate_sharpe_ratio,
                                                      rebalance_penalty=rebalance_penalty,
                                                      reward_measure=reward_measure,
                                                      step_version="paper")])
    elif settings.STRATEGY_MODE == "ppoCustomBase":
        env = FinancialMarketEnv(df=df,
                                 features_list=features_list,
                                 single_features_list=single_features_list,
                                 day=day,
                                 iteration=iteration,
                                 # only used for logger.info
                                 model_name=strategy_mode,
                                 mode=mode,
                                 hmax_normalize=hmax_normalize,
                                 initial_cash_balance=initial_cash_balance,
                                 transaction_fee_percent=transaction_fee,
                                 reward_scaling=reward_scaling,
                                 assets_dim=assets_dim,
                                 shape_observation_space=shape_observation_space,
                                 initial=initial,
                                 # for validation, we always have an "initial state"
                                 previous_state=previous_state,
                                 previous_asset_price=previous_asset_price,
                                 price_colname=price_column_name,
                                 results_dir=results_dir,
                                 reset_counter=reset_counter,
                                 logger=logger,
                                 save_results=save_results,
                                 calculate_sharpe_ratio=calculate_sharpe_ratio,
                                 performance_calculation_window=7,
                                 step_version=env_step_version,
                                 rebalance_penalty=rebalance_penalty,
                                 reward_measure=reward_measure
                                 )
    else:
        print("ERROR - no valid strategy mode passed. cannot create instance env_train.")
    return env


def get_model(train_environment,
              number_train_data_points: int,
              shape_observation_space: int,
              assets_dim: int,
              strategy_mode: str,
              env_step_version: str = "",
              validation_environment=None,
              train_env_firstday: int= 0,
              val_env_firstday: int=0,
              load_trained_model: bool=False,
              trained_model_save_path: str=None,
              current_episode_number: int=None,
              performance_save_path=None,
              logger=None,
              now_hptuning=False,
              use_tuned_params=False,
              gamma=None,
              gae_lambda=None,
              clip_epsilon=None,
              critic_loss_coef=None,
              entropy_loss_coef=None,
              net_version=None,
              optimizer=None,
              optimizer_learning_rate=None,
              max_gradient_norm=None,
              total_timesteps_to_collect=None,
              num_epochs=None,
              batch_size=None,
              ):

    if strategy_mode == "ppoCustomBase":
        # if we use my custom PPO algorithm, we first need to create an instance of some of the components,
        # namely the "brain" and the "buffer", before passing them to the PPO algorithm together with the environment
        brain = BrainActorCritic(init_weights_actor_net=init_weights_actor_net,
                                 init_weights_critic_net=init_weights_critic_net,
                                 init_weights_feature_extractor_net=init_weights_feature_extractor_net,
                                 observations_size=shape_observation_space,
                                 # size of observations AFTER going through the env; the env also adds positions for n. asset holdings
                                 actions_num=assets_dim,
                                 version=net_version,
                                 env_step_version=env_step_version,
                                 optimizer=optimizer,
                                 learning_rate=optimizer_learning_rate,
                                 )
        if load_trained_model and current_episode_number > 1:
            # loading trained model to brain
            logger.info("loading saved model from train_model_save_path.")
            brain.load_state_dict(torch.load(trained_model_save_path))

        logger.info("initialized  /loaded -Brain- parameters: ")
        #for param in brain.parameters():
        #    logger.info(param)
        # NOTE: the buffer size is the number of available training points (because these we want to store)
        # but we also want to update the model in mini batches, e..g of size 84 (by default),
        # and then the buffer size should b a multiple of the batch size because otherwise the last data batch,
        # which is not as long as batch:size, would be ignored.
        # instead, the last batch is just going to have some zero padding at the end so that it will still be of the same length
        #buffer_size = max(number_train_data_points, math.ceil(number_train_data_points / batch_size) * batch_size)
        buffer_size=number_train_data_points
        logger.info(f"buffer size:  {buffer_size}")
        logger.info(f"number of train data points passed: {number_train_data_points}")
        buffer = OnPolicyBuffer(buffer_size=buffer_size,
                                obs_shape=(shape_observation_space,),
                                actions_number=assets_dim)
        if now_hptuning or use_tuned_params:
            if now_hptuning:
                logger.info("taking hyperparameters for tuning to model.")
            if use_tuned_params:
                logger.info("using tuned hyperparameters to train the model.")
            gamma = gamma
            gae_lambda = gae_lambda
            clip_epsilon = clip_epsilon
            critic_loss_coef = critic_loss_coef
            entropy_loss_coef = entropy_loss_coef
        else:
            gamma = gamma
            gae_lambda = gae_lambda
            clip_epsilon = clip_epsilon
            critic_loss_coef = critic_loss_coef
            entropy_loss_coef = entropy_loss_coef
            logger.info("CHECK parameters")
            logger.info(f"GAMMA             : {gamma}"
                        f"GAE_LAMBDA        : {gae_lambda}"
                        f"CLIP_EPSILON      : {clip_epsilon}"
                        f"CRITIC_LOSS_COEF  : {critic_loss_coef}"
                        f"ENTROPY_LOSS_COEF : {entropy_loss_coef}"
                        )
        model = PPO_algorithm(env_train=train_environment,
                              env_validation=validation_environment,
                              brain=brain,
                              buffer=buffer,
                              batch_size=batch_size, # note: batch size is changed if train data is < batch_size
                              num_epochs=num_epochs,
                              gamma=gamma,
                              gae_lambda=gae_lambda,
                              clip_epsilon=clip_epsilon,
                              critic_loss_coef=critic_loss_coef,
                              entropy_loss_coef=entropy_loss_coef,
                              max_gradient_normalization=max_gradient_norm,
                              total_timesteps_to_collect=total_timesteps_to_collect,
                              performance_save_path=performance_save_path,
                              current_episode=current_episode_number,
                              train_env_firstday=train_env_firstday,
                              val_env_firstday=val_env_firstday,
                              logger=logger,
                              env_step_version=env_step_version,
                              assets_dim=assets_dim
                              )

    elif strategy_mode == "ppo": # todo: rm
        if load_trained_model and current_episode_number > 1:
            model = PPO.load(trained_model_save_path)
            model.set_env(train_environment)
        else:
            if now_hptuning:
                logger.info("taking hyperparameters for tuning to model.")
                gamma = gamma
                gae_lambda = gae_lambda
                clip_epsilon = clip_epsilon
                critic_loss_coef = critic_loss_coef
                entropy_loss_coef = entropy_loss_coef
            else:
                gamma = agent_params.ppo.GAMMA
                gae_lambda = agent_params.ppo.GAE_LAMBDA
                clip_epsilon = agent_params.ppo.CLIP_RANGE
                critic_loss_coef = agent_params.ppo.VF_COEF
                entropy_loss_coef = agent_params.ppo.ENT_COEF
                logger.info("CHECK parameters")
                logger.info(f"GAMMA             : {gamma}"
                            f"GAE_LAMBDA        : {gae_lambda}"
                            f"CLIP_EPSILON      : {clip_epsilon}"
                            f"CRITIC_LOSS_COEF  : {critic_loss_coef}"
                            f"ENTROPY_LOSS_COEF : {entropy_loss_coef}"
                            )
            # if we use the stable baselines3 version of PPO, we import it like this:
            model = PPO(policy='MlpPolicy',  # was PPO2 in sb2 tf1.x
                        env=train_environment,  # environment where the agent learns and acts
                        # parameters directly imported from config.py
                        seed=settings.SEED,
                        ent_coef=entropy_loss_coef,
                        learning_rate=agent_params.ppo.LEARNING_RATE,
                        n_steps=agent_params.ppo.N_STEPS,
                        n_epochs=agent_params.ppo.N_EPOCHS,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        clip_range=clip_epsilon,
                        vf_coef=critic_loss_coef,
                        clip_range_vf=agent_params.ppo.CLIP_RANGE_VF,
                        max_grad_norm=agent_params.ppo.MAX_GRAD_NORM,
                        use_sde=agent_params.ppo.USE_SDE,
                        sde_sample_freq=agent_params.ppo.SDE_SAMPLE_FREQ,
                        target_kl=agent_params.ppo.TARGET_KL,
                        create_eval_env=agent_params.ppo.CREATE_EVAL_ENV,
                        policy_kwargs=agent_params.ppo.POLICY_KWARGS,
                        verbose=agent_params.ppo.VERBOSE,
                        device=agent_params.ppo.DEVICE,
                        _init_setup_model=agent_params.ppo.INIT_SETUP_MODEL)
    return model

def hyperparameter_tuning(train_data: pd.DataFrame, # todo: rm
                          assets_dim: int,
                          shape_observation_space: int,
                          results_dir: str,
                          trained_model_save_path: str,
                          validation_folds: int = 4,
                          logger: logging = None,
                          save_results: bool = True,
                          ) -> list:
    """
    This function implements time series cross validation for hyperparameter tuning
    on the first training subset of the whole time series data available.
    """
    # create directory for hptuning in the current run
    hp_tuning_results_dir = os.path.join(results_dir, "hp_tuning")
    os.makedirs(hp_tuning_results_dir)

    # get all hyperparameter combinations to try:
    hp = pd.read_csv(os.path.join("data", "hp_grid.csv"), index_col=0)
    hp_dict = {}
    for col in list(hp.columns.values):
        hp_dict.update({f"combination_{col}": hp[col]})

    #n = 0
    for key in list(hp_dict.keys()):
        if save_results:
            hp_tuning_results_keydir = os.path.join(hp_tuning_results_dir, str(key))
            os.makedirs(f"{hp_tuning_results_keydir}/buy_trades")
            os.makedirs(f"{hp_tuning_results_keydir}/cash_value")
            os.makedirs(f"{hp_tuning_results_keydir}/datadates")
            os.makedirs(f"{hp_tuning_results_keydir}/exercised_actions")
            os.makedirs(f"{hp_tuning_results_keydir}/last_state")
            os.makedirs(f"{hp_tuning_results_keydir}/number_asset_holdings")
            os.makedirs(f"{hp_tuning_results_keydir}/policy_actions")
            os.makedirs(f"{hp_tuning_results_keydir}/portfolio_value")
            os.makedirs(f"{hp_tuning_results_keydir}/rewards")
            os.makedirs(f"{hp_tuning_results_keydir}/sell_trades")
            os.makedirs(f"{hp_tuning_results_keydir}/state_memory")
            os.makedirs(f"{hp_tuning_results_keydir}/transaction_cost")
            os.makedirs(f"{hp_tuning_results_keydir}/all_weights_cashAtEnd")
            os.makedirs(f"{hp_tuning_results_keydir}/asset_equity_weights")

        logger.info(f"Start HP Tuning: {key}")
        logger.info(list(hp.index.values))
        logger.info(hp_dict[key])

        # get current hyperparameter combination
        gamma, gae_lam, clip, critic_loss_coef, entropy_loss_coef = hp_dict[key]

        # first year is only train data
        #base_train_data = train_data.iloc[0:365, :]
        #base_train_data_length = len(base_train_data.index)
        rest_train_data_length = len(train_data[train_data.index >= train_data.index[0] + 365].index.unique())
        # only the rest train data is split into validation sets, e.g. in 4 (by default)
        validation_days_per_split = math.ceil(rest_train_data_length / validation_folds) # last validation set wil be a bit smaller, if the ratio is not "round"

        sharpe_ratios_dict = {}
        sharpe_ratios_list_one_hp_combi = []
        for i in range(0, validation_folds):
            # take a subset of the whole train set
            train_subset = train_data[train_data.index <= train_data.index[0] + 365 + (validation_days_per_split*i)]
            #if n == 0:
                #print(train_subset)

            # create the train environment for the hptuning for the i'th csval fold
            env_train = get_environment(df=train_subset,
                                        day=train_subset.index[0],
                                        iteration=i,  # only used for logger.info
                                        mode="train",
                                        assets_dim=assets_dim,
                                        shape_observation_space=shape_observation_space,
                                        initial=True,
                                        previous_state=[],
                                        previous_asset_price=[],
                                        results_dir=hp_tuning_results_keydir,
                                        reset_counter=0,
                                        logger=logger,
                                        save_results=save_results,
                                        calculate_sharpe_ratio=False
                                        )
            # seed the environment and the action space
            env_train.seed(settings.SEED)
            env_train.action_space.seed(settings.SEED)

            # get the validation data and create the validation env
            validation_subset = train_data[(train_data.index >= train_subset.index[-1]) &
                                           (train_data.index <= train_subset.index[-1] + validation_days_per_split)]
            #if n == 0:
                #print(validation_subset)

            env_val = get_environment(df=validation_subset,
                                      day=validation_subset.index[0],
                                      iteration=i,  # only used for logger.info
                                      mode="validation",
                                      assets_dim=assets_dim,
                                      shape_observation_space=shape_observation_space,
                                      initial=True,
                                      previous_state=[],
                                      previous_asset_price=[],
                                      results_dir=hp_tuning_results_keydir,
                                      reset_counter=0,
                                      logger=logger,
                                      save_results=save_results,
                                      calculate_sharpe_ratio=True # ! we will use the annualized daily sharpe ratio as performance metric
                                      )
            # seed the environment and the action space
            obs_val = env_val.reset()
            env_val.seed(settings.SEED)
            env_val.action_space.seed(settings.SEED)

            # get the model
            ppo_model = get_model(train_environment=env_train,
                                  validation_environment=None,
                                  number_train_data_points=len(train_data.index.unique()),
                                  shape_observation_space=shape_observation_space,
                                  assets_dim=assets_dim,
                                  strategy_mode=settings.STRATEGY_MODE,
                                  performance_save_path=None,#os.path.join(results_dir, "training_performance"),
                                  train_env_firstday=train_data.index[0],
                                  val_env_firstday=None,
                                  load_trained_model=False,
                                  trained_model_save_path=trained_model_save_path,
                                  current_episode_number=i,
                                  logger=logger,
                                  # parameter only actove for hp tuning
                                  now_hptuning=True,
                                  gamma=gamma,
                                  gae_lambda=gae_lam,
                                  clip_epsilon=clip,
                                  critic_loss_coef=critic_loss_coef,
                                  entropy_loss_coef=entropy_loss_coef
                                  )
            training_timesteps = len(train_data.index.unique()) * agent_params.ppoCustomBase.TOTAL_EPISODES_TO_TRAIN
            # train the model
            ppo_model.learn(total_timesteps=training_timesteps)
            # calculate the performance

            logger.info("-Brain- parameters before validation of hptuning: ")
            for param in ppo_model.Brain.parameters():
                logger.info(param)

            while True:
                # use the trained model to predict actions using the test_obs we received far above when we setup the test env
                # and run obs_test = env.reset()
                action, _ = ppo_model.predict(obs_val)
                # take a step in the test environment and get the new test observation, reward, dones (a mask if terminal state True or False)
                # and info (here empty, hence _, since we don't need it)
                obs_val, rewards, dones, annualized_sharpe_ratio = env_val.step(action)
                if dones:
                    # get performance metrics
                    # save SR
                    sharpe_ratios_list_one_hp_combi.append(annualized_sharpe_ratio)
                    break
        # update median sharpe ratio for current key
        #n += 1
        #print("sharpe ratios list")
        #print(sharpe_ratios_list_one_hp_combi)
        median_sharpe = np.median(sharpe_ratios_list_one_hp_combi)
        sharpe_ratios_dict.update({str(key): median_sharpe})
    # get key with max. sharpe ratio
    best_hp_combi_key = max(sharpe_ratios_dict, key=sharpe_ratios_dict.get)

    #get the best combination from the best key:
    gamma, gae_lam, clip, critic_loss_coef, entropy_loss_coef = hp_dict[best_hp_combi_key]

    params_df = pd.DataFrame({"parameter": list(hp.index.values),
                              "best_combi": [gamma, gae_lam, clip, critic_loss_coef, entropy_loss_coef]})
    params_df["sharpe_ratio"] = sharpe_ratios_dict[best_hp_combi_key]
    params_df.to_csv(os.path.join(hp_tuning_results_dir,"best_parameter_combination.csv"))

    logger.info(f"--Finished HP-Tuning on train subset.")
    logger.info("Best Parameter Combination:")
    logger.info(f"GAMMA             : {gamma}")
    logger.info(f"GAE_LAMBDA        : {gae_lam}")
    logger.info(f"CLIP_EPSILON      : {clip}")
    logger.info(f"CRITIC_LOSS_COEF  : {critic_loss_coef}")
    logger.info(f"ENTROPY_LOSS_COEF : {entropy_loss_coef}")

    return [gamma, gae_lam, clip, critic_loss_coef, entropy_loss_coef]