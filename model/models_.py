import time
import logging
# RL models from stable-baselines
from stable_baselines3 import PPO  # PPO2
from stable_baselines3 import A2C
from stable_baselines3 import DDPG

# from stable_baselines3.ddpg.policies import DDPGPolicy
# from stable_baselines3.ppo.policies import MlpPolicy #, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise  # , AdaptiveParamNoiseSpec
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.gail import generate_expert_traj, ExpertDataset  # TODO: check if they did this somewhere (?)
from config.config import paths, crisis_settings, settings, env_params, agent_params, dataprep_settings
from preprocessing.preprocessors import *
# customized env
# from env.EnvMultipleStock_train import StockEnvTrain
from env.FinancialMarketEnv import FinancialMarketEnv  # StockEnvValidation
# from env.EnvMultipleStock_trade import StockEnvTrade
import os

# set seeds:


#######################################################################
# DEFINE FUNCTIONS FOR TRAINING, PREDICTION AND PERORMANCE EVALUATION #
#######################################################################

def DRL_train(env_train,
              trained_dir,
              results_dir,
              save_name,
              iteration,
              agent_name=settings.STRATEGY_MODE,
              load_trained_model=False,
              trained_model_save_path=None,
              ):
    """

    @param env_train:
    @param trained_dir:
    @param save_name:
    @param agent_name:
    @param timesteps:
    @return:
    """
    start = time.time()
    if agent_name == "ppo":
        if load_trained_model:
            timesteps = agent_params._ppo.TRAINING_TIMESTEPS // 100
            model = PPO.load(trained_model_save_path)
            model.set_env(env_train)
            #logging.warning("get PPO parameters:")
            #logging.warning(PPO.get_parameters())
        else:
            timesteps = agent_params._ppo.TRAINING_TIMESTEPS
            model = PPO(policy='MlpPolicy',  # was PPO2 in sb2 tf1.x
                        env=env_train,  # environment where the agent learns and acts
                        seed=settings.SEED_AGENT,
                        ent_coef=agent_params._ppo.ent_coef,  # entropy coefficient
                        # default parameters (unchanged from default as given by stable-baselines)
                        learning_rate=agent_params._ppo.learning_rate,  # can also be variable, e.g. a function of the current progress remaining etc.
                        # todo: where is learning rate used exactly?
                        # nminibatches=8, # todo: was in old paper
                        n_steps=agent_params._ppo.n_steps,
                        n_epochs=agent_params._ppo.n_epochs, #todo: surrogate loss?
                        gamma=agent_params._ppo.gamma, # todo: would taking an actual interest rate make sense?
                        gae_lambda=agent_params._ppo.gae_lambda, # todo: ?
                        clip_range=agent_params._ppo.clip_range,
                        clip_range_vf=agent_params._ppo.clip_range_vf, # todo: ?
                        vf_coef=agent_params._ppo.vf_coef,
                        max_grad_norm=agent_params._ppo.max_grad_norm,
                        use_sde=agent_params._ppo.use_sde, # todo: ?
                        sde_sample_freq=agent_params._ppo.sde_sample_freq,
                        target_kl=agent_params._ppo.target_kl,
                        tensorboard_log=agent_params._ppo.tensorboard_log,
                        create_eval_env=agent_params._ppo.create_eval_env,
                        policy_kwargs=agent_params._ppo.policy_kwargs,
                        verbose=agent_params._ppo.verbose,
                        device=agent_params._ppo.device,
                        _init_setup_model=agent_params._ppo.init_setup_model)
    elif agent_name == "a2c":
        if load_trained_model:
            timesteps = agent_params._a2c.TRAINING_TIMESTEPS // 10
            model = A2C.load(trained_model_save_path)
            model.set_env(env_train)
        else:
            timesteps = agent_params._a2c.TRAINING_TIMESTEPS
            model = A2C(policy='MlpPolicy',
                        env=env_train,
                        verbose=0,
                        seed=settings.SEED_AGENT,
                        # default values (given by stable-baselines):
                        )
    elif agent_name == "ddpg":
        if load_trained_model:
            timesteps = agent_params._ddpg.TRAINING_TIMESTEPS // 10
            model = DDPG.load(trained_model_save_path)
            model.set_env(env_train)
        else:
            timesteps = agent_params._ddpg.TRAINING_TIMESTEPS
            # add the noise objects for DDPG
            n_actions = env_train.action_space.shape[-1]
            #param_noise = None
            if agent_params._ddpg.action_noise == "OUAN": # todo: check if formula below makes sense, why done so and if too hardcoded
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            else:
                action_noise = None
            model = DDPG(policy='MlpPolicy',
                         env=env_train,
                         seed=settings.SEED_AGENT,
                         #param_noise=param_noise, # todo: was in paper version, was None
                         action_noise=action_noise, # default: None
                         # default values (given by stable-baselines):
                         learning_rate=agent_params._ddpg.learning_rate,
                         buffer_size=agent_params._ddpg.buffer_size,
                         learning_starts=agent_params._ddpg.learning_starts,
                         batch_size=agent_params._ddpg.batch_size,
                         tau=agent_params._ddpg.tau,
                         gamma=agent_params._ddpg.gamma,
                         train_freq=agent_params._ddpg.train_freq,
                         gradient_steps=agent_params._ddpg.gradient_steps,
                         optimize_memory_usage=agent_params._ddpg.optimize_memory_usage,
                         tensorboard_log=agent_params._ddpg.tensorboard_log,
                         create_eval_env=agent_params._ddpg.create_eval_env,
                         policy_kwargs=agent_params._ddpg.policy_kwargs,
                         verbose=agent_params._ddpg.verbose,
                         device=agent_params._ddpg.device,
                         _init_setup_model=agent_params._ddpg.init_setup_model)
    else:
        print("ERROR (DRL_train) - provided agent_name does not exist.")

    model.learn(total_timesteps=timesteps)
    end = time.time()

    save_name = f"{agent_name}_timesteps_{timesteps}_" + save_name
    trained_model_save_path = f"{trained_dir}/{save_name}"
    model.save(trained_model_save_path)
    logging.warning(f"Training time ({agent_name.upper()}): " + str((end - start) / 60) + " minutes.")

    last_state_flattened, last_state, _, _, _ = env_train.render()
    df_last_state = pd.DataFrame({'last_state': last_state_flattened})
    df_last_state.to_csv(f'{results_dir}/last_state/last_state_train_{agent_name}_i{iteration}.csv',
                         index=False)
    return model, last_state, trained_model_save_path  # returns trained model


# TODO: (changed from DL_prediction to DRL_trading)
def DRL_predict(trained_model,
                test_data,
                test_env,
                test_obs,
                mode,
                iteration,
                model_name,
                results_dir):  # "validation"

    start = time.time()
    for j in range(len(test_data.index.unique())):
        action, _states = trained_model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)
        if mode == "trade" and j == (len(test_data.index.unique()) - 2):  # todo: understand
            last_state_flattened, last_state, _, _, _ = test_env.render()
            df_last_state = pd.DataFrame({'last_state': last_state_flattened})
            df_last_state.to_csv(f'{results_dir}/last_state/last_state_{mode}_{model_name}_i{iteration}.csv',
                                 index=False)
    end = time.time()
    logging.warning(f"{mode} time: " + str((end - start) / 60) + " minutes")

    if mode == "validation":
        return None
    elif mode == "trade":
        return last_state


def get_performance_metrics(model_name,
                            iteration,
                            results_dir):
    # RENAME IN get_performance_metrics and add more metrics # todo: was get_validation_sharpe
    ###Calculate Sharpe ratio based on validation results which were saved (see validation env) ###
    df_total_value = pd.read_csv('{}/portfolio_value/portfolio_value_train_{}_i{}.csv'.format(
        results_dir, model_name, iteration), index_col=0)
    df_total_value['daily_return'] = df_total_value["portfolio_value"].pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe
