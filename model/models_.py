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
            timesteps = agent_params._ppo.TRAINING_TIMESTEPS // 4
            model = PPO.load(trained_model_save_path)
            model.set_env(env_train)
            #logging.warning("get PPO parameters:")
            #logging.warning(PPO.get_parameters())
        else:
            timesteps = agent_params._ppo.TRAINING_TIMESTEPS
            model = PPO(policy='MlpPolicy',  # was PPO2 in sb2 tf1.x
                        env=env_train,  # environment where the agent learns and acts
                        seed=settings.SEED_AGENT,
                        ent_coef=agent_params._ppo.ENT_COEF,  # entropy coefficient
                        # default parameters (unchanged from default as given by stable-baselines)
                        learning_rate=agent_params._ppo.LEARNING_RATE,  # can also be variable, e.g. a function of the current progress remaining etc.
                        # todo: where is learning rate used exactly?
                        # nminibatches=8, # todo: was in old paper
                        n_steps=agent_params._ppo.N_STEPS,
                        n_epochs=agent_params._ppo.N_EPOCHS, #todo: surrogate loss?
                        gamma=agent_params._ppo.GAMMA, # todo: would taking an actual interest rate make sense?
                        gae_lambda=agent_params._ppo.GAE_LAMBDA, # todo: ?
                        clip_range=agent_params._ppo.CLIP_RANGE,
                        clip_range_vf=agent_params._ppo.CLIP_RANGE_VF, # todo: ?
                        vf_coef=agent_params._ppo.VF_COEF,
                        max_grad_norm=agent_params._ppo.MAX_GRAD_NORM,
                        use_sde=agent_params._ppo.USE_SDE, # todo: ?
                        sde_sample_freq=agent_params._ppo.SDE_SAMPLE_FREQ,
                        target_kl=agent_params._ppo.TARGET_KL,
                        tensorboard_log=agent_params._ppo.TENSORBOARD_LOG,
                        create_eval_env=agent_params._ppo.CREATE_EVAL_ENV,
                        policy_kwargs=agent_params._ppo.POLICY_KWARGS,
                        verbose=agent_params._ppo.VERBOSE,
                        device=agent_params._ppo.DEVICE,
                        _init_setup_model=agent_params._ppo.INIT_SETUP_MODEL)
    elif agent_name == "a2c":
        if load_trained_model:
            timesteps = agent_params._a2c.TRAINING_TIMESTEPS // 4
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
            timesteps = agent_params._ddpg.TRAINING_TIMESTEPS // 4
            model = DDPG.load(trained_model_save_path)
            model.set_env(env_train)
        else:
            timesteps = agent_params._ddpg.TRAINING_TIMESTEPS
            # add the noise objects for DDPG
            n_actions = env_train.action_space.shape[-1]
            #param_noise = None
            if agent_params._ddpg.ACTION_NOISE == "OUAN": # todo: check if formula below makes sense, why done so and if too hardcoded
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            else:
                action_noise = None
            model = DDPG(policy='MlpPolicy',
                         env=env_train,
                         seed=settings.SEED_AGENT,
                         #param_noise=param_noise, # todo: was in paper version, was None
                         action_noise=action_noise, # default: None
                         # default values (given by stable-baselines):
                         learning_rate=agent_params._ddpg.LEARNING_RATE,
                         buffer_size=agent_params._ddpg.BUFFER_SIZE,
                         learning_starts=agent_params._ddpg.LEARNING_STARTS,
                         batch_size=agent_params._ddpg.BATCH_SIZE,
                         tau=agent_params._ddpg.TAU,
                         gamma=agent_params._ddpg.GAMMA,
                         train_freq=agent_params._ddpg.TRAIN_FREQ,
                         gradient_steps=agent_params._ddpg.GRADIENT_STEPS,
                         optimize_memory_usage=agent_params._ddpg.OPTIMIZE_MEMORY_USAGE,
                         tensorboard_log=agent_params._ddpg.TENSORBOARD_LOG,
                         create_eval_env=agent_params._ddpg.CREATE_EVAL_ENV,
                         policy_kwargs=agent_params._ddpg.POLICY_KWARGS,
                         verbose=agent_params._ddpg.VERBOSE,
                         device=agent_params._ddpg.DEVICE,
                         _init_setup_model=agent_params._ddpg.INIT_SETUP_MODEL)
    else:
        print("ERROR (DRL_train) - provided agent_name does not exist.")

    model.learn(total_timesteps=timesteps)
    end = time.time()

    save_name = f"{agent_name}_timesteps_{timesteps}_" + save_name
    trained_model_save_path = f"{trained_dir}/{save_name}"
    model.save(trained_model_save_path)
    logging.warning(f"Training time ({agent_name.upper()}): " + str((end - start) / 60) + " minutes.")

    last_state_flattened, last_state, last_asset_price, _, _, _ = env_train.render()
    df_last_state = pd.DataFrame({'last_state': last_state_flattened})
    df_last_state.to_csv(f'{results_dir}/last_state/last_state_train_{agent_name}_i{iteration}.csv',
                         index=False)
    return model, last_state, last_asset_price, trained_model_save_path  # returns trained model


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
    act=[]
    for j in range(len(test_data.index.unique())):
        action, _states = trained_model.predict(test_obs)

        act.append(action[0])


        test_obs, rewards, dones, info = test_env.step(action)
        if mode == "trade" and j == (len(test_data.index.unique()) - 2):  # todo: understand
            last_state_flattened, last_state, last_price, _, _, _ = test_env.render()
            df_last_state = pd.DataFrame({'last_state': last_state_flattened})
            df_last_state.to_csv(f'{results_dir}/last_state/last_state_{mode}_{model_name}_i{iteration}.csv',
                                 index=False)
    end = time.time()
    logging.warning(f"{mode} time: " + str((end - start) / 60) + " minutes")

    #print("action shape: ", action.shape)
    #print("action: ", action)
    #print("action type: ", type(action))
    #print("act list: ", act)
    #print("act shape: ", len(act))
    #print("act type: ", type(act))

    #li = [i * 100 for i in list(range(0, len(act)))]
    #lidf = pd.DataFrame({"li": li})
    #actdf = pd.DataFrame(act)
    #pd.concat([lidf, actdf], axis=1).to_csv("action.csv")
    #pd.DataFrame(list(act)).to_csv("actionlist.csv")
    #pd.DataFrame(act).to_json("action.json")

    if mode == "validation":
        return None
    elif mode == "trade":
        return [last_state, last_price]


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
