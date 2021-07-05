import time
# RL models from stable-baselines
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import gym

# own libraries
import logging
from config.config import *
from environment.FinancialMarketEnv import FinancialMarketEnv
from model.CustomOnPolicyBuffer import OnPolicyBuffer
from model.CustomPPOAlgorithm import PPO_algorithm
from model.CustomActorCriticNets import BrainActorCritic, \
    init_weights_feature_extractor_net, init_weights_actor_net, init_weights_critic_net

########################################################################
# DEFINE FUNCTIONS FOR TRAINING, PREDICTION AND PERFORMANCE EVALUATION #
########################################################################

def get_model(train_environment,
              number_train_data_points: int,
              shape_observation_space: int,
              assets_dim: int,
              performance_save_path: str,
              strategy_mode: str=settings.STRATEGY_MODE,
              validation_environment=None,
              train_env_firstday: int=0,
              val_env_firstday: int=0,
              load_trained_model: bool=False,
              trained_model_save_path: str=None):

    if strategy_mode == "ppoCustomBase":
        # if we use my custom PPO algorithm, we first need to create an instance of some of the components,
        # namely the "brain" and the "buffer", before passing them to the PPO algorithm together with the environment
        brain = BrainActorCritic(init_weights_actor_net=init_weights_actor_net,
                                 init_weights_critic_net=init_weights_critic_net,
                                 init_weights_feature_extractor_net=init_weights_feature_extractor_net,
                                 observations_size=shape_observation_space,
                                 # size of observations AFTER going through the env; the env also adds positions for n. asset holdings
                                 actions_num=assets_dim,
                                 version=agent_params.ppoCustomBase.NET_VERSION,
                                 optimizer=agent_params.ppoCustomBase.OPTIMIZER,
                                 learning_rate=agent_params.ppoCustomBase.OPTIMIZER_LEARNING_RATE,
                                 )
        if load_trained_model:
            # loading trained model to brain
            brain.load_state_dict(torch.load(trained_model_save_path))
        buffer = OnPolicyBuffer(buffer_size=number_train_data_points,
                                obs_shape=(shape_observation_space,),
                                actions_number=assets_dim)
        model = PPO_algorithm(env_train=train_environment,
                              env_validation=validation_environment,
                              brain=brain,
                              buffer=buffer,
                              batch_size=agent_params.ppoCustomBase.BATCH_SIZE,
                              num_epochs=agent_params.ppoCustomBase.NUM_EPOCHS,
                              gamma=agent_params.ppoCustomBase.GAMMA,
                              gae_lambda=agent_params.ppoCustomBase.GAE_LAMBDA,
                              clip_epsilon=agent_params.ppoCustomBase.CLIP_EPSILON,
                              max_kl_value=agent_params.ppoCustomBase.MAX_KL_VALUE,
                              critic_loss_coef=agent_params.ppoCustomBase.CRITIC_LOSS_COEF,
                              entropy_loss_coef=agent_params.ppoCustomBase.ENTROPY_LOSS_COEF,
                              max_gradient_normalization=agent_params.ppoCustomBase.MAX_GRADIENT_NORMALIZATION,
                              total_timesteps_to_collect=agent_params.ppoCustomBase.TOTAL_TIMESTEPS_TO_COLLECT,
                              performance_save_path=performance_save_path,
                              train_env_firstday=train_env_firstday,
                              val_env_firstday=val_env_firstday)
    elif strategy_mode == "ppo":
        if load_trained_model:
            model = PPO.load(trained_model_save_path)
            model.set_env(train_environment)
        else:
            # if we use the stable baselines3 version of PPO, we import it like this:
            model = PPO(policy='MlpPolicy',  # was PPO2 in sb2 tf1.x
                        env=train_environment,  # environment where the agent learns and acts
                        # parameters directly imported from config.py
                        seed=settings.SEED,
                        ent_coef=agent_params.ppo.ENT_COEF,
                        learning_rate=agent_params.ppo.LEARNING_RATE,
                        n_steps=agent_params.ppo.N_STEPS,
                        n_epochs=agent_params.ppo.N_EPOCHS,
                        gamma=agent_params.ppo.GAMMA,
                        gae_lambda=agent_params.ppo.GAE_LAMBDA,
                        clip_range=agent_params.ppo.CLIP_RANGE,
                        vf_coef=agent_params.ppo.VF_COEF,
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