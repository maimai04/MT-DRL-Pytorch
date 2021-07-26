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
from model.CustomActorCriticNets import *

########################################################################
# DEFINE FUNCTION FOR TRAINING                                         #
########################################################################

def get_model(train_environment,
              number_train_data_points: int,
              shape_observation_space: int,
              shape_lstm_observation_space: int,
              assets_dim: int,
              env_step_version: str = "",
              validation_environment=None,
              train_env_firstday: int= 0,
              val_env_firstday: int=0,
              load_trained_model: bool=False,
              trained_model_save_path: str=None,
              current_episode_number: int=None,
              performance_save_path=None,
              logger=None,
              gamma=None,
              gae_lambda=None,
              clip_epsilon=None,
              critic_loss_coef=None,
              entropy_loss_coef=None,
              net_arch=None,
              optimizer=None,
              optimizer_learning_rate=None,
              max_gradient_norm=None,
              total_timesteps_to_collect=None,
              num_epochs=None,
              batch_size=None,
              predict_deterministic=False,
              ):

    # if we use my custom PPO algorithm, we first need to create an instance of some of the components,
    # namely the "brain" and the "buffer", before passing them to the PPO algorithm together with the environment

    #### CREATE INSTANCE OF ACTOR-CRITIC NETWORK, CALLED BRAIN (naming: since it is like a brain of the agent...haha)
    brain = BrainActorCritic(init_weights_actor_net=init_weights_actor_net,
                             init_weights_critic_net=init_weights_critic_net,
                             init_weights_feature_extractor_net=init_weights_feature_extractor_net,
                             observations_size=shape_observation_space,
                             # size of observations AFTER going through the env; the env also adds positions for n. asset holdings
                             actions_num=assets_dim,
                             net_arch=net_arch,
                             env_step_version=env_step_version,
                             optimizer=optimizer,
                             learning_rate=optimizer_learning_rate,
                             lstm_observations_size=shape_lstm_observation_space,
                             lstm_hidden_size_feature_extractor=32,
                             lstm_num_layers=2,
                             feature_extractor_class=FeatureExtractorNet,
                             actor_class=ActorNet,
                             critic_class=CriticNet,
                             )
    # if we have the setting that we want to initialize the weights based on a trained model
    # and we are not in the first episode (else we have no previously trained model):
    if load_trained_model and current_episode_number > 1:
        # loading trained model to brain
        logger.info("loading saved model from train_model_save_path:")
        # load trained model to Brain instance
        brain.load_state_dict(torch.load(trained_model_save_path))
        # log the brain parameters to the logging file for debugging
        for name, param in brain.named_parameters():
            if param.requires_grad:
                logger.info(f"name: {name}")
                logger.info(param.grad)

    logger.info("initialized -Brain-: ")
    logger.info(brain)

    #### CREATE INSTANCE FOR THE BUFFER WHICH STORES THE COLLECTED TRAJECTORIES
    # NOTE: the buffer size is the number of available training points (because these we want to store)
    # that also means the last batch will probably not be of length batch_size, since the data is not an exact multiple of the batch_size
    buffer_size = number_train_data_points
    logger.info(f"buffer size:  {buffer_size}")
    logger.info(f"number of train data points passed: {number_train_data_points}")
    # here, we change the asset dimension for the ppo agent only, if we want to predict an additional action (portfolio weight)
    # for the cash position as well (only relevant for the Dirichlet policy version)
    if env_step_version == "newNoShort2":
        assets_dim_buffer = assets_dim + 1 # one for cash
    else:
        assets_dim_buffer = assets_dim
    buffer = OnPolicyBuffer(buffer_size=buffer_size,
                            obs_shape=(shape_observation_space,),
                            lstm_obs_shape=(shape_lstm_observation_space,),
                            actions_number=assets_dim_buffer,
                            )

    ##### CREATE INSTANCE FOR THE PPO AGENT ALGORITHM
    # for that, we need to pass it the previously created brain and buffer
    model = PPO_algorithm(env_train=train_environment,
                          env_validation=validation_environment,
                          brain=brain,
                          buffer=buffer,
                          batch_size=batch_size,
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
                          assets_dim=assets_dim,
                          predict_deterministic=predict_deterministic,
                          )

    return model
