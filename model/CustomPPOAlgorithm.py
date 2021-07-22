import math

import numpy as np
from torch import nn
import pandas as pd
import os
import torch


########################################################################
# CUSTOM PPO ALGORITHM                                                 #
########################################################################

class PPO_algorithm():
    """
    This class implements the Proximal Policy Optimization (PPO) algorithm based on
    the paper: https://arxiv.org/abs/1707.06347 of Schulman et al (2017), which is from people from OpenAI.
    The details of the implementation are based mainly on the following sources, apart from the abovementioned paper:
     - https://arxiv.org/abs/1506.02438 for generalized advantage estimation
     - https://spinningup.openai.com/en/latest/algorithms/ppo.html for detailed documentation about PPO
     - https://stable-baselines.readthedocs.io/en/master/ Stable Baselines 1 (tensorflow 1.x version)
     - https://stable-baselines3.readthedocs.io/en/master/ Stable Baselines 3 (Pytorch version)

    """

    def __init__(self,
                 # classes we need to pass for agent construction
                 env_train,
                 brain,
                 buffer,
                 current_episode: int,
                 # number of assets, needed for the scaling function
                 assets_dim: int,
                 # optional validation env for e.g. early stopping based on average reward
                 env_validation=None,
                 # params for neural network
                 batch_size: int=64,
                 num_epochs: int=10,
                 # discount rate for rewards
                 gamma: float=0.99,
                 # smoothing value for generalized advantage estimators (exponential mean discounting factor)
                 gae_lambda: float=0.95,
                 # clipping value for surrogate loss of actor (policy)
                 clip_epsilon: float=0.2,
                 # max. kullback leibner value allowed. if None: kl not used as metric for early stopping
                 max_kl_value: float=None,
                 # critic (value) loss coefficient (how much weight does the value loss have in the total loss)
                 critic_loss_coef: float=0.5,
                 # entropy loss coefficient (how much weight does netropy have in the total loss)
                 entropy_loss_coef: float=0.01,
                 # maximal gradient normalization
                 max_gradient_normalization: float=0.5,
                 # total timesteps of experiences to be sampled from the environment with current policy
                 # and stored in the Buffer. Only relevant if < number of total available time steps
                 # in training dataset, else the collection of eyperiences will simply stop when
                 # we have reached the end of the data set.
                 total_timesteps_to_collect: int=5000,
                 train_env_firstday: int=0,
                 val_env_firstday: int=0,
                 logger=None,
                 # path where loss curves etc. are saved
                 performance_save_path: str = None,
                 env_step_version: str = "paper",
                 predict_deterministic: bool=False,
                 ):
        """
        Here, the variables and hyperparameters are initialized.
        """
        self.logger = logger
        ### initialize classes we need for PPO agent construction and their parameters
        self.Env = env_train
        self.Env_firstday = train_env_firstday
        self.EnvVal = env_validation
        self.EnvVal_firstday = val_env_firstday
        self.env_step_version = env_step_version
        self.assets_dim = assets_dim
        self.Brain = brain
        self.OnPolicyBuffer = buffer

        # NOTE: make sure that:
        # buffer_size = math.ceil(total_timesteps_to_collect/batch_size)* batch_size,
        # so that the buffer will include all equal batches, with the last batch having zero padding.
        # also make sure that total_timesteps_to_collect = len(train_data)
        self.buffer_size = self.OnPolicyBuffer.buffer_size
        ### initialize hyperparameters
        # network parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_gradient_normalization = max_gradient_normalization
        # rl agent-specific parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.max_kl_value = max_kl_value
        self.critic_loss_coef = critic_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        # total training steps of experience we want to collect to the buffer storage
        self.total_timesteps_to_collect = total_timesteps_to_collect

        self.performance_save_path = performance_save_path
        self.current_episode = current_episode

        self.predict_deterministic = predict_deterministic

    def _collect_experiences_to_buffer(self, total_timesteps_to_collect):
        """
        When this function is called, the ppo agent collects experience (trajectories)
        in the environment (env) that we have passed to the agent above.
        The environment contains the training data we have passed to the environment
        and in this function, we go over the whole data once, until we have sampled
        actions at every time step in the training data.

        The observations, actions, action log probabilities, rewards, dones (whether
        we have reached the last observation (True) or not (False), only True for when
        we reached the end of the training set)), rewards_to_go (future discounted rewards
        at each time step)).
        """
        # get first state / observation from the environment by resetting it
        self.logger.info("env_firstday: "+str(self.Env_firstday))

        # get first observation from training environment
        obs, lstm_obs = self.Env.reset(day=self.Env_firstday)#, initial=True)

        #print("(PPO) lstm obs: ")
        #print(lstm_obs)
        #print("length")
        #print(len(lstm_obs))
        # finally, we need to initialize a hidden state in a certain shape, let's call it lstm_state
        # see also: https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
        # Initialize hidden state and cells state as (h, c), with zeros, for actor and critic separately
        # Note: here we collect steps, so sequence length = 1
        if self.Brain.net_arch == "mlplstm_separate":
            lstm_state = None # no shared lstm state, only one separate for actor and a separate one for critic
            lstm_state_actor = self.Brain.feature_extractor_actor.create_initial_lstm_state(sequence_length=1)
            lstm_state_critic = self.Brain.feature_extractor_critic.create_initial_lstm_state(sequence_length=1)
        elif self.Brain.net_arch == "mlplstm_shared":
            lstm_state = self.Brain.feature_extractor.create_initial_lstm_state(sequence_length=1)
            lstm_state_actor = None # no separate states, only share lstm state between actor and critic
            lstm_state_critic = None
        # on lstm and truncated BPTT: https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
        else: # if no lstm used, lstm states are put to None in order to avoid erorrs later. They will have no effect
            lstm_state = None
            lstm_state_actor = None
            lstm_state_critic = None

        # scale the observation (see documentation at the bottom for function "scale_observations"
        obs = self.scale_observations(obs=obs, env_step_version=self.env_step_version, n_assets=self.assets_dim)
        #print("scaled_obs: ")
        #print(obs)

        #self.logger.info("train env reset, first obs: ")
        #self.logger.info(obs[0:100]) # to check if the observations are correct
        #self.logger.info("data ", self.Env.data)

        # reset the Buffer in order to empty storage from previously collected trajectories
        self.OnPolicyBuffer.reset()
        # self.logger.info("first observations: ")
        # self.logger.info(obs)

        # we start at step 0 in our replay buffer experience collection
        current_timesteps_collected = 0

        # we fill the Replay buffer with trajectories (state, action, reward, next state)
        # resp. state = observation / obs here
        # on the way we also do a forward pass on each data point through the
        # actor and critic networks in order to obtain action & action log probability
        # from te actor and the state-value estimate from the critic.

        ##### STARTING COLLECTING TRAJECTORIES AND SAVING THEM INTO THE BUFFER #####
        while current_timesteps_collected < total_timesteps_to_collect:

            #---------------------#
            ### 1. FORWARD PASS ###
            #---------------------#
            # of current observation based on OLD policy (no updates yet in this module)
            # torch.no_grad() is used to set gradient computation to False. It reduces memory consumption.
            # it can be used here because we will only collect trajectories, and not do backpropagation here,
            # so we don't need the automatical gradient computation
            # see also: https://pytorch.org/docs/stable/generated/torch.no_grad.html
            with torch.no_grad():
                # convert obs (np.array) to torch tensor, so it can be used by the neural network
                obs = torch.as_tensor(obs, dtype=torch.float)
                lstm_obs = torch.as_tensor(lstm_obs, dtype=torch.float)
                # get value estimate, sampled action and actions log probabilities from brain,
                # which uses actor and critic architecture
                # Note: if we don't use the lstm, the lstm states are simply None. Else, they are tensors that are going to be used
                # in the next iteration as input
                # output: value estimate (V_estimate) given current observations (& lstm observations, if applicable) from critic (value network)
                #         predicted actions from actor (policy network), log probabilities of actions, _(action means), _(actions_distribution),
                #         standard deviation (for saving to csv to check if allright),
                #         last hidden and cell state of lstm (lstm_state), same for actor and critic
                V_estimate, actions, actions_log_prob, _, _, stdev, lstm_state_new, lstm_state_actor_new, lstm_state_critic_new \
                    = self.Brain(observations=obs,
                                evaluation_mode=False, # only forward pass to get actions, no evaluation of actions
                                actions=None, # no actions because only forward pass
                                actions_deterministic=False, # we want exploration during training and sample actions from a distribution
                                # below only applicable if lstm arch chosen (see config.py), else will not have effect (will be None)
                                lstm_observations=lstm_obs,
                                lstm_states=lstm_state,
                                lstm_states_actor=lstm_state_actor,
                                lstm_states_critic=lstm_state_critic)
                # Note: this yields a value estimate for the current observation (state) => one value, as tensor of dim (1,)
                # actions for the current obs (state), e.g. one weight for each stock, as tensor of dim (n_assets,)
                # action log probabilities, for the current obs (state), one for each stock action, as tensor of dim (n_assets,)

            # convert actions to numpy array so it can be used by the environment to get the
            # take a step, get the next state and compute the reward
            # note: must use .detach() on torch.tensors before converting them to numpy because it yields en error otherwise
            # why: because we cannot call numpy on a tensor that requires gradient (requires_grad = True)
            actions = actions.numpy()#.detach().numpy() # .cpu()

            # ------------------------------------------------------------#
            ### 2. TAKE A STEP IN THE ENVIRONMENT USING SAMPLED ACTIONS ###
            # ------------------------------------------------------------#
            # in order to obtain the new state, a reward and a mask (done; True if
            # end of trajectory (dataset) reached, else False))
            new_obs, new_lstm_obs, reward, done, _ = self.Env.step(actions) # note: actions need to be scaled / clipped before usage, this is done in the env
            # scale the observations
            new_obs = self.scale_observations(obs=new_obs, env_step_version=self.env_step_version, n_assets=self.assets_dim)

            # --------------------------------#
            ### 3. SAVE OBTAINED TRAJECTORY ###
            # --------------------------------#
            # add obtained experience data to OnPolicyBuffer storage
            self.OnPolicyBuffer.add(obs, "obs", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(actions, "actions", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(actions_log_prob, "actions_log_probs", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(V_estimate, "value_estimates", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(reward, "rewards", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(done, "dones", position=current_timesteps_collected)
            # add lstm observations
            self.OnPolicyBuffer.add(lstm_obs, "lstm_obs", position=current_timesteps_collected)

            # only applicable if we are using lstm with separate layers for actor and critic
            # we have to detach from graph otherwise we will use it in backpropagation and we don't want that (would be calculation intense
            # and the idea her is to have truncated BPTT)
            if self.Brain.net_arch == "mlplstm_separate":
                lstm_state = None
                # set the next lstm states as the ones we just received above from Brain
                # again, need to detach, see also: https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
                lstm_state_actor = (lstm_state_actor[0].detach(), lstm_state_actor[1].detach())
                lstm_state_critic = (lstm_state_critic[0].detach(), lstm_state_critic[1].detach())
                # add the states to the buffer
                self.OnPolicyBuffer.add(lstm_state_actor[0], "lstm_state_actor_h", position=current_timesteps_collected)
                self.OnPolicyBuffer.add(lstm_state_actor[1], "lstm_state_actor_c", position=current_timesteps_collected)
                self.OnPolicyBuffer.add(lstm_state_critic[0], "lstm_state_critic_h", position=current_timesteps_collected)
                self.OnPolicyBuffer.add(lstm_state_critic[1], "lstm_state_critic_c", position=current_timesteps_collected)
            elif self.Brain.net_arch == "mlplstm_shared":
                lstm_state_actor = None
                lstm_state_critic = None
                lstm_state = (lstm_state_new[0].detach(), lstm_state_new[1].detach())
                self.OnPolicyBuffer.add(lstm_state[0], "lstm_state_h", position=current_timesteps_collected)
                self.OnPolicyBuffer.add(lstm_state[1], "lstm_state_c", position=current_timesteps_collected)


            # the new observation becomes the "current observation" of the next iteration
            # unless the new observation is the last state; the last state (and the last state value)
            # will not be saved into the buffer, because we don't need them for training later
            # (for example: a forward pass on the last observation in order to obtain new action probabilities to compare with the old
            # action probabilities, is useless, since we have not sampled an action for the last observation. There is no step coming after
            # the last observation.)
            obs = new_obs
            lstm_obs = new_lstm_obs
            # collect trajectories step by step until total_rb_timesteps are reached
            current_timesteps_collected += 1

            # ---------------------------------------------------#
            ### 4a. REPEAT UNTIL ENOUGH TRAJECTORIES COLLECTED ###
            # ---------------------------------------------------#
            # if done = True, we have reached the end of our data set and we break the
            # loop. This applies if total_timesteps_to_collect have been set higher than the actually
            # available time steps in the data set
            if current_timesteps_collected in list(range(0, total_timesteps_to_collect, 1000)) + \
                    [total_timesteps_to_collect-1]:
                self.logger.info(f"current timesteps collected: {current_timesteps_collected + 1} / max. {total_timesteps_to_collect}")
            if done:
                self.logger.info("experience collection finished (because episode finished  (done)). ")
                break
        # -------------------------------------------------------------------------------------------#
        ### 4b. WHEN DONE: GET ESTIMATE FOR TERMINAL VALUE TO CALCULATE RETURNS AND SAVE TO BUFFER ###
        ###     BY DOING A FORWARD PASS                                                            ###
        # -------------------------------------------------------------------------------------------#
        # now we need to get the value estimates for the terminal state, the new_obs
        # we need the terminal value estimate in order tp compute the advantages below
        with torch.no_grad():
            obs = torch.as_tensor(new_obs, dtype=torch.float)
            lstm_obs = torch.as_tensor(new_lstm_obs, dtype=torch.float)
            V_terminal_estimate, _, _, _, _, stdev, _, _, _ \
                = self.Brain(observations=obs,
                             evaluation_mode=False, # we are only doing a forward pass to get the latest state value, no evaluation of actions
                             actions=None, # no evaluation of actions, hence no actions passed
                             actions_deterministic=False,
                             # only applicable if we are using an lstm, else None
                             lstm_observations=lstm_obs,
                             lstm_states=lstm_state,
                             lstm_states_actor=lstm_state_actor,
                             lstm_states_critic=lstm_state_critic,
                             )
        self.OnPolicyBuffer.calculate_and_store_advantages(terminal_V_estimate=V_terminal_estimate,
                                                           gamma=self.gamma,
                                                           gae_lambda=self.gae_lambda)
        self.OnPolicyBuffer.calculate_and_store_returns()

    def learn(self,
              total_timesteps: int=100000, # by default, but the real value is set in run_pipeline.py
              batch_size: int = 64):
        """
        This method implements the actual learning process of the reinforcement learning
        agent, including the training of the actor-critic functions (e.g. neural networks)

        We use the collected data from collect_experiences_to_rollout_buffer()
        and learn from it:
          we do a forward pass in order to get value estimates from the critic
          for each state (note: theoretically we could also do this when we sample actions
          using a forward pass through the actor network in the collect_experiences_to_rollout_buffer()
          function, but theoretically the step belongs more to this function, because
          we didn't need the values in the previous step in order to sample trajectories / experiences,
          and sampling trajectories / experiences is the sole purpose of
          the collect... function call.)
        """
        total_learning_timesteps = total_timesteps
        # NOTE ON: total time steps to learn on;
        # e.g. if we have collected 1000 timesteps from our env & data,
        # and we want to learn on total 2000 time steps, then we will learn on the same data we have
        # twice (we will go twice over the data we have, and that for multiple epochs
        # (so actually more than twice if num_epochs>1))
        # this parameter should not be too large => over-training, not too small => under-training
        learning_timesteps_done = 0
        timestep_start_last_batch = (total_learning_timesteps // batch_size) * batch_size + 1
        last_batch = False
        # we do this until we have learned for total_learning_timesteps

        # for saving later
        suffix = "epoch_"
        epoch_colnames = [suffix + str(s) for s in list(range(1, self.num_epochs + 1))]

        # at the end of every 10 epochs, validation rewards are calculated on the holdout data set
        explained_variances_whole_training = []
        actor_loss_whole_training = []
        critic_loss_whole_training = []
        entropy_loss_whole_training = []
        combined_loss_whole_training = []

        # ------------------#
        ### LEARNING LOOP ###
        # ------------------#

        # Note: we have our total_learning_timesteps set. As long as we haven't learned for this much of timesteps,
        # we are again and again, sampling new trajectories into the rollout buffer and then training on them, then
        # again sampling new trajectories and then again training on them etc. until we have trained for enough time steps.
        while learning_timesteps_done < total_learning_timesteps:
            # ----------------------------------------------------------------------#
            ### 1. COLLECT TRAJECTORIES / EXPERIENCES AND STORE IN ROLLOUT BUFFER ###
            # ----------------------------------------------------------------------#
            # collect experience in the environment based on the current policy and store in buffer
            self._collect_experiences_to_buffer(total_timesteps_to_collect=self.total_timesteps_to_collect)

            ### CALCULATE EXPLAINED VARIANCE
            # calculate the explained variance (current value estimates and returns (= target value))
            # if >1: value estimates from critic are a good predictor of (bootstrapped) returns
            # if <0: worse than predicting nothing (in the beginning, this is what we get)
            # see: https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2
            # The next time trajectories are going to be collected, hopefully the estimates are going to be more accurate so that
            # the explained variance is going to go up because the estimation error goes down (or the variance in the returns goes up, but then probably the
            # estimation error would also go up because probably, the returns and values would converge more frequently, but this is just an assumption).
            # see : https://www.oreilly.com/library/view/machine-learning-algorithms/9781789347999/49c4b96f-a567-4513-8e91-9f41b0b4dab0.xhtml
            all_returns = self.OnPolicyBuffer.trajectory_dict["returns"].flatten()
            all_V_estimates = self.OnPolicyBuffer.trajectory_dict["value_estimates"].flatten()
            variance_target_value = np.var(all_returns)
            variance_estimation_error = np.var(all_returns - all_V_estimates)
            explained_variance = 1 - variance_estimation_error / variance_target_value
            explained_variances_whole_training.append(explained_variance)
            del all_returns, all_V_estimates, variance_target_value, variance_estimation_error, explained_variance

            ### INITIALIZE EMPTY LISTS FOR STORING TRAINING LOSSES AND OTHER STUFF
            # initialize emtpy lists for storing parameters / performance metrics over epochs
            # every batch parameter is going to be appended to these lists
            prob_ratio_all_epochs = []
            surr_loss_1_all_epochs = []
            surr_loss_2_all_epochs = []
            surr_loss_all_epochs = []
            actor_loss_all_epochs = []
            critic_loss_all_epochs = []
            entropy_loss_all_epochs = []
            combined_loss_all_epochs = []
            advantages_all_epochs = []
            returns_all_epochs = []
            standard_deviations_all_epochs = []
            action_means_all_epochs = []
            value_estimates_all_epochs = []

            # ----------------------------------------------------------------------#
            ### 2. TRAINING: CALCULATE LOSSES OF MINIBATCHES, UPDATE NETWORKS     ###
            ###              FOR EACH EPOCH                                       ###
            # ----------------------------------------------------------------------#
            # now we train for multiple epochs
            for epoch in range(1, self.num_epochs + 1):
                self.logger.info(f"---EPOCH: {epoch} / {self.num_epochs}")

                ### CREATE EMPTY LISTS TO STORE RESULTS FOR EACH EPOCH
                # every batch parameter is going to be appended to these lists
                prob_ratio_of_epoch = []
                surr_loss_1_of_epoch = []
                surr_loss_2_of_epoch = []
                surr_loss_of_epoch = []
                actor_loss_of_epoch = []
                critic_loss_of_epoch = []
                entropy_loss_of_epoch = []
                combined_loss_of_epoch = []
                advantages_of_epoch = []
                returns_of_epoch = []
                standard_deviations_of_epoch = []
                action_means_of_epoch = []
                value_estimates_of_epoch = []

                # ---------------------------------#
                ### 2a. TRAINING: FOR EACH BATCH ###
                # ---------------------------------#
                # get batch from each tensor of batches
                # Note: normally, "batch" means the whole training data, here
                # with "batch" I actually ean "minibatch", but I write "batch" because it is shorter
                start_idx = 0
                for batch_num, batch_idx in enumerate(range(self.batch_size, self.buffer_size+1, self.batch_size)):
                    # note: must convert to torch tensor because otherwise error
                    batch_obs = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["obs"][start_idx:batch_idx], dtype=torch.float)
                    batch_actions = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["actions"][start_idx:batch_idx], dtype=torch.float)
                    batch_actions_log_probs = torch.tensor(self.OnPolicyBuffer.trajectory_dict["actions_log_probs"][start_idx:batch_idx].flatten(), dtype=torch.float)
                    #batch_rewards = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["rewards"][start_idx:batch_idx].flatten(), dtype=torch.float)
                    batch_returns = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["returns"][start_idx:batch_idx].flatten(), dtype=torch.float)
                    #batch_V_estimates = torch.tensor(self.OnPolicyBuffer.trajectory_dict["value_estimates"][start_idx:batch_idx])  #
                    batch_advantages = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["advantage_estimates"][start_idx:batch_idx].flatten(), dtype=torch.float)  #
                    batch_lstm_obs = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["lstm_obs"][start_idx:batch_idx], dtype=torch.float)


                    if self.Brain.net_arch == "mlplstm_separate":
                        first_lstm_state_actor_h = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["lstm_state_actor_h"][start_idx], dtype=torch.float)
                        first_lstm_state_actor_c = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["lstm_state_actor_c"][start_idx], dtype=torch.float)
                        first_lstm_state_critic_h = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["lstm_state_critic_h"][start_idx], dtype=torch.float)
                        first_lstm_state_critic_c = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["lstm_state_critic_c"][start_idx], dtype=torch.float)
                        first_lstm_state_actor = (first_lstm_state_actor_h, first_lstm_state_actor_c)
                        first_lstm_state_critic = (first_lstm_state_critic_h, first_lstm_state_critic_c)
                        first_lstm_state = None
                        # Note: we only need the first lstm state of each batch
                        # (= lstm state which encodes all previous time series before the start of this batch)
                        #print("first_lstm_state_actor")
                        #print(first_lstm_state_actor)
                        #print("first_lstm_state_critic")
                        #print(first_lstm_state_critic)
                    elif self.Brain.net_arch == "mlplstm_shared":
                        first_lstm_state_h = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["lstm_state_h"][start_idx], dtype=torch.float)
                        first_lstm_state_c = torch.as_tensor(self.OnPolicyBuffer.trajectory_dict["lstm_state_c"][start_idx], dtype=torch.float)
                        first_lstm_state = (first_lstm_state_h, first_lstm_state_c)
                        first_lstm_state_actor = None
                        first_lstm_state_critic = None
                        # Note: we only need the first lstm state of each batch
                        # (= lstm state which encodes all previous time series before the start of this batch)
                        #print("first_lstm_state")
                        #print(first_lstm_state)
                    else:
                        first_lstm_state=None
                        first_lstm_state_actor=None
                        first_lstm_state_critic=None

                    ### UPDATE START INDEX BY BATCH SIZE
                    start_idx += self.batch_size

                    #if batch_num in [0, math.ceil(self.buffer_size/self.batch_size)]:
                    #    self.logger.info(f"BATCH NUMBER: {batch_num + 1}")
                    #if epoch == 1 and start_idx == 0:
                        #self.logger.info(f"sample batch (observations), (len: {len(batch_obs)})")
                        #self.logger.info(batch_obs)

                    # -----------------------------------------------------------------------------#
                    ### 2b. EVALUATION OF OLD ACTIONS AND CALCULATIMG LOSSES                     ###
                    # -----------------------------------------------------------------------------#
                    # Standardizing advantage estimates (creating z-score) across each batch
                    # (not in the paper, but discussed here for example, in the comments: https://openreview.net/forum?id=r1etN1rtPB)
                    # many implementations use a normalized version because it makes convergence
                    # more stable and also decreases the variance of our advantage estimates
                    # many people say that it therefore makes learning more stable
                    # at the end, a very small number s added (e.g. 1e-100) so that we never divide by 0
                    advantage_est_standardized = (batch_advantages - batch_advantages.mean()) / \
                                                 (batch_advantages.std() + 1e-100)

                    # convert the actions from the batch (old actions, since these are actions with the "old" policy, the one
                    # we used to sample trajectories into the buffer) to torch tensor, since we are going to use them so
                    # sample the new action log probabilities
                    old_actions = batch_actions
                    # Get value estimates, current action log probabilities and action distribution entropy
                    # with current weights from critic doing a forward pass
                    # NOTE: in the first round the forward pass will yield the same actions / action distr. / action log probs.
                    # as when we collected trajectories into the OnPolicyBuffer, hence the probabilities ratio (calculated later) will be 1
                    # NOTE: this time we should not do the forward pass with "with torch.no_grad()", because we will later call backward()
                    # and want to be able to compute the gradients in order to update our policy

                    #print("batch_obs: ")
                    #print(batch_obs)
                    new_V_estimate, current_action_new_log_prob, action_distr_entropy, action_means, _, stdev = \
                        self.Brain(observations=batch_obs, #torch.as_tensor(batch_obs, dtype=torch.float),
                                   evaluation_mode=True, # this time, we evaluate old actions
                                   actions=old_actions,  # old actions are going to be evaluated
                                   # only applicable if lstm used, else None
                                   lstm_observations=batch_lstm_obs,
                                   lstm_states=first_lstm_state,
                                   lstm_states_actor=first_lstm_state_actor,
                                   lstm_states_critic=first_lstm_state_critic)  # evaluation mode must be true: see documentation in Brain class
                    # Calculate the probabilities ratio (since we have log probabilities, we can simply
                    # subtract them in the exponential function and we get back the "original", non-log probabilities)
                    # log probabilities are used as a convention, because they make calculations easier
                    # see also: https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # note: the log probabilities are scalars, hence the probability ratio will be a scalar value as well, not a vector of ratios
                    proba_ratio = torch.exp(current_action_new_log_prob - batch_actions_log_probs)
                    # in the first iteration (before first policy update),
                    # the ratio should be 1, since we will sample the same actions with the same net weights every time
                    ###  ACTOR LOSS (POLICY LOSS, SURROGATE LOSS)
                    # r * A
                    surr_loss_1 = proba_ratio * advantage_est_standardized
                    # this can be: negative, if adv. negative, else positive

                    # clipped_r * A
                    surr_loss_2 = torch.clamp(proba_ratio,
                                              min=1 - self.clip_epsilon,
                                              max=1 + self.clip_epsilon) * advantage_est_standardized
                    # this can be negative if the advantage is negative,
                    # but it will be equal or smaller than surr_loss1, because it is clipped

                    # Note: because we will use gradient descent, not gradient Ascent, we need to take the negative of the surrogate function
                    # (in the paper, they maximize the (non-negative) surrogate function / loss with Gradient ascent)
                    surr_loss = torch.min(surr_loss_1, surr_loss_2)

                    # calculate the clipped surrogate los (=actor loss / policy network loss)
                    # since the actor loss in the paper is actually defined as "actor gain", so we would have to maximize it with
                    # gradient ascent; but here we want to use gradient descent so we will take the negative of the surrogate loss mean
                    actor_loss = -torch.mean(surr_loss)

                    ### CRITIC LOSS (VALUE LOSS)
                    # Value loss using the TD(gae_lambda) target in the calculation of the returns (hence more smooth, less variance)
                    # it is the mean squared error between the estimated state value V and the actual returns (smoothed)
                    # we need to flatten the new value estimates, because its a tensor like this: [[v1],[v1], ...]
                    # and we need it to be like this: [v1,v2,...] (rewards are also like this), else we et an error below
                    new_V_estimate = new_V_estimate.flatten()
                    # Note: since critic loss = MSE loss (mean squared error), it is always going to be positive
                    critic_loss = nn.functional.mse_loss(new_V_estimate, batch_returns)  # returns used as target value

                    ### ENTROPY LOSS
                    #self.logger.info("action distr. entropy before backprop:")
                    #self.logger.info(action_distr_entropy)
                    entropy_loss = -torch.mean(action_distr_entropy)

                    # TOTAL LOSS FUNCTION:
                    # Note: the total loss for gradient ascent would be: actor_loss(=gain) - c1*critic_loss(=loss) + c2*entropy_loss(=gain, the higher the better),
                    # but since we do gradient descent, actor_loss and entropy loss are negative, value loss is positive (opposite signs)
                    # note: the negative sign for the actor loss has already been added some lines above when we took the negative of the
                    # so it is then: total actor loss: max a-c+e = min -a+c-e <= this we use
                    # surrogate function, similarly for the entropy loss above
                    total_loss = actor_loss + self.critic_loss_coef * critic_loss + self.entropy_loss_coef * entropy_loss

                    # -----------------------------------------#
                    ### 2c. UPDATE BRAIN (WHOLE NETWORK(s) ) ###
                    # -----------------------------------------#
                    # UPDATING THE ACTOR-CRITIC MODEL (Updates policy, feature extractor and value network)
                    # Note: we first need to call zero_grad() on the optimizer, in order to clear all the gradients from the previous iteration,
                    # because otherwise we would be accumulating gradients over multiple iterations (batches) for the update, which would not be good
                    # then we use loss.backward() to backpropagate the loss and compute the current gradients (derivatives w.r.t. the parameters) for the current batch
                    # then we use a gradient clipping method in order to make sure that our gradients don't "explode"
                    # see also: https://www.reddit.com/r/MachineLearning/comments/4qshk3/max_norm_gradientweight_clipping_for/
                    # then we use optimizer.step(), with this we take a step in the direction of the gradient: this is the actual update step, where our weights are updated
                    # see also: https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301
                    self.Brain.optimizer.zero_grad()
                    total_loss.backward()
                    # avoiding exploding gradients with gradient normalization:
                    # https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/
                    # https://www.reddit.com/r/MachineLearning/comments/4qshk3/max_norm_gradientweight_clipping_for/
                    # default value the same as the one from stable baselines for reproducibility purposes
                    torch.nn.utils.clip_grad_norm_(self.Brain.parameters(), self.max_gradient_normalization)

                    # every now and then, log gradients to check if they are ok (shouldn't be flat out zero,
                    # at least not most of the time)
                    #for name, param in self.Brain.named_parameters():
                    #    if param.requires_grad:
                     #       print(f"name: {name}")
                    #        print(param.grad)
                    if epoch == 3 and learning_timesteps_done==0 and batch_num==1:
                        self.logger.info(f"(epoch: {epoch}, Brain gradients before optimizer step:")
                        for name, param in self.Brain.named_parameters():
                            if param.requires_grad:
                                self.logger.info(f"name: {name}")
                                self.logger.info(param.grad)
                    # now, finally, we take ONE step in the direction of the gradient
                    self.Brain.optimizer.step()

                    # AT THE END OF EACH BATCH:
                    # append metrics
                    # metrics that come as a multi-dim tensor per batch (one entry per observation in batch)
                    # each of length batch_size (e.g. 64)
                    # => list of lists => flatten list so that we get only a list of values, one for each obs
                    #if learning_timesteps_done >= timestep_start_last_batch:#total_learning_timesteps-64:
                    prob_ratio_of_epoch.append(np.array(proba_ratio.detach()).flatten())
                    surr_loss_1_of_epoch.append(np.array(surr_loss_1.detach()).flatten())
                    surr_loss_2_of_epoch.append(np.array(surr_loss_2.detach()).flatten())
                    surr_loss_of_epoch.append(np.array(surr_loss.detach()).flatten())
                    advantages_of_epoch.append(np.array(advantage_est_standardized.detach()).flatten())
                    returns_of_epoch.append(np.array(batch_returns.detach()).flatten())
                    try: # because if we use dirichlet distr., the tensor here is Nan (no std. estimated)
                        standard_deviations_of_epoch.append(np.array(stdev.detach()).flatten())
                    except:
                        standard_deviations_of_epoch.append(np.array(stdev).flatten())
                    action_means_of_epoch.append(np.array(action_means.detach()).flatten())
                    value_estimates_of_epoch.append(np.array(new_V_estimate.detach()).flatten())
                    # metrics that come as single values per batch (each of length 1)
                    # => list of values
                    actor_loss_of_epoch.append(actor_loss.detach())
                    critic_loss_of_epoch.append(critic_loss.detach())
                    entropy_loss_of_epoch.append(entropy_loss.detach())
                    combined_loss_of_epoch.append(total_loss.detach())


                    if epoch == self.num_epochs:
                        self.logger.info(f"epoch {epoch}.")
                        self.logger.info(f"learning timesteps before update (total to do: {total_learning_timesteps}):")
                        self.logger.info(learning_timesteps_done)

                        # update learning timesteps only for last (or any, just only one) epoch,
                        # because we only want to count the steps we train on (not how often we trained on them)
                        learning_timesteps_done += self.batch_size

                        self.logger.info("learning timesteps after update:")
                        self.logger.info(learning_timesteps_done)

                # AT THE END OF ALL BATCHES (ONE EPOCH)
                #self.logger.info(f"Avg loss per epoch: {np.mean(epoch_loss)}")
                # after each epoch end, append
                # (each below will then be a list of lists)
                # => list (for all epochs together) of lists (one for each epoch) of values (one for each obs in batch, e.g. 64)
                # [[v,v,v],[v,v,v],[v,v,v],...] = [epoch1, epoch2, ...]
                # a vector or array per day
                if learning_timesteps_done >=timestep_start_last_batch: #total_learning_timesteps - 64:
                    advantages_all_epochs.append(np.array(advantages_of_epoch).flatten())
                    returns_all_epochs.append(np.array(returns_of_epoch).flatten())
                    prob_ratio_all_epochs.append(np.array(prob_ratio_of_epoch).flatten())
                    surr_loss_1_all_epochs.append(np.array(surr_loss_1_of_epoch).flatten())
                    surr_loss_2_all_epochs.append(np.array(surr_loss_2_of_epoch).flatten())
                    surr_loss_all_epochs.append(np.array(surr_loss_of_epoch).flatten())
                    standard_deviations_all_epochs.append(np.array(standard_deviations_of_epoch).flatten()) # one std for all actions for each day (64 for a batch)
                    action_means_all_epochs.append(np.array(action_means_of_epoch).flatten()) # at each day, n_asset number of actions
                    value_estimates_all_epochs.append(np.array(value_estimates_of_epoch).flatten())

                # one loss per batch
                actor_loss_all_epochs.append(actor_loss_of_epoch)
                critic_loss_all_epochs.append(critic_loss_of_epoch)
                entropy_loss_all_epochs.append(entropy_loss_of_epoch)
                combined_loss_all_epochs.append(combined_loss_of_epoch)

                self.logger.info(f"---EPOCH: {epoch} / {self.num_epochs} done.")
                self.logger.info(f"total Epoch loss: {np.sum(combined_loss_of_epoch)}")
                self.logger.info(f"average total Epoch loss: {np.mean(combined_loss_of_epoch)}")

            # AT THE END OF ALL EPOCHS
            actor_loss_whole_training.append(actor_loss_all_epochs)
            critic_loss_whole_training.append(critic_loss_all_epochs)
            entropy_loss_whole_training.append(entropy_loss_all_epochs)
            combined_loss_whole_training.append(combined_loss_all_epochs)

            # after all 10 epochs (or else, as defined in config.py), we save the data to .csv
            if self.performance_save_path != None and learning_timesteps_done >= timestep_start_last_batch:#total_learning_timesteps-64:
                # save arrays to separate .csv files, for which there is an estimate per STEP
                for li, liname in zip([advantages_all_epochs, returns_all_epochs, prob_ratio_all_epochs,
                                       surr_loss_1_all_epochs, surr_loss_2_all_epochs, surr_loss_all_epochs,
                                       standard_deviations_all_epochs, action_means_all_epochs, value_estimates_all_epochs],
                                      ["advantages_all_epochs", "returns_all_epochs", "prob_ratio_all_epochs",
                                       "surr_loss_1_all_epochs", "surr_loss_2_all_epochs", "surr_loss_all_epochs",
                                       "standard_deviations_all_epochs", "action_means_all_epochs", "value_estimates_all_epochs"]):
                    pd.DataFrame(np.transpose(li)).to_csv(os.path.join(self.performance_save_path,
                                                                             f"{liname}_"
                                                                             f"ep{self.current_episode}_"
                                                                             "LearningTimestepsDone_"
                                                                             f"{learning_timesteps_done}.csv"))
                pd.DataFrame({"explained_variance": explained_variances_whole_training}).to_csv(
                    os.path.join(self.performance_save_path,
                                 f"explained_variance_"
                                 f"ep{self.current_episode}_"
                                 "LearningTimestepsDone_"
                                 f"{learning_timesteps_done}.csv"))
                # save the relevant losses to one combined .csv file, for which there is one estimate per UDPATE
                # (which in my work, is per one batch of 64 steps)
                pd.DataFrame({"actor_loss": np.array(actor_loss_whole_training).flatten(),
                              "critic_loss": np.array(critic_loss_whole_training).flatten(),
                              "entropy_loss": np.array(entropy_loss_whole_training).flatten(),
                              "combined_loss": np.array(combined_loss_whole_training).flatten(),
                              }).to_csv(os.path.join(self.performance_save_path,
                                                     f"train_performances_"
                                                     f"ep{self.current_episode}_"
                                                     f"LearningTimestepsDone_{learning_timesteps_done}.csv"))

            if learning_timesteps_done >= total_learning_timesteps-64:
                self.logger.info("-Brain- parameters after training: ")
                for param in self.Brain.parameters():
                    self.logger.info(param)

            # after all 10 epochs, if we have passed a validation env, we do some "out of sample testing"
            # this is for monitoring how well we perform in training and how well we generalize on the validation set
            if self.EnvVal is not None:
                self.logger.info(f"Validation beginning.")
                self._validation(learning_timesteps_done=learning_timesteps_done,
                                 total_learning_timesteps_todo=total_learning_timesteps)
                self.logger.info(f"Validation ended.")

    def _validation(self, learning_timesteps_done=None, total_learning_timesteps_todo=None) -> None:
        validation_rewards = []
        validation_values = []
        obs_val, lstm_obs_val = self.EnvVal.reset(day=self.EnvVal_firstday)
        # lstm stuff
        lstm_state = None # note: the validation set does not get any initial lstm state
        lstm_state_actor = None
        lstm_state_critic = None
        # here, obs_val and lstm_obs_val are always only 1 step, hence sequence_length = 1
        if self.Brain.net_arch == "mlplstm_separate":
            lstm_state = None
            lstm_state_actor = self.Brain.feature_extractor_actor.create_initial_lstm_state(sequence_length=1)
            lstm_state_critic = self.Brain.feature_extractor_critic.create_initial_lstm_state(sequence_length=1)
        elif self.Brain.net_arch == "mlplstm_shared":
            lstm_state = self.Brain.feature_extractor.create_initial_lstm_state(sequence_length=1)
            lstm_state_actor = None
            lstm_state_critic = None

        # do validation in a loop until done (reached the end of the data set)
        for j in range(len(self.EnvVal.df.index.unique())):
            # note: scaling is already included in the predict function, see below,
            # hence we do not need to scale obs before prediction here
            val_actions, val_value, lstm_state, lstm_state_actor, lstm_state_critic = \
                self.predict(new_obs=obs_val,
                             env_step_version=self.env_step_version,
                             n_assets=self.assets_dim,
                             # only if lstm used, else None
                             new_lstm_obs=lstm_obs_val,
                             lstm_state=lstm_state,
                             lstm_state_actor=lstm_state_actor,
                             lstm_state_critic=lstm_state_critic,
                             predict_deterministic=self.predict_deterministic)
            obs_val, lstm_obs_val, val_rewards, val_done, _ = self.EnvVal.step(val_actions)
            # append validation rewards to list
            validation_rewards.append(val_rewards) # note: validation rewards are saved to .csv via env already
            validation_values.append(val_rewards)
            if val_done:
                break
        self.logger.info(f"validation mean reward: {np.mean(validation_rewards)}")
        self.logger.info(f"validation total reward: {np.mean(validation_rewards)}")

        if self.performance_save_path != None and learning_timesteps_done >= total_learning_timesteps_todo-64:
            # save array for value estimates in validation set to separate .csv files, for which there is an estimate per STEP
            pd.DataFrame({"value_estimates_validation": np.array(validation_values)}).to_csv(os.path.join(self.performance_save_path,
                                                                     f"value_estimates_validation_"
                                                                     f"ep{self.current_episode}_"
                                                                     "LearningTimestepsDone_"
                                                                     f"{learning_timesteps_done}.csv"))
        return None # Note: rewards are already saved to csv by the validation env

    def predict(self, new_obs, env_step_version, n_assets,
                new_lstm_obs=None, lstm_state=None, lstm_state_actor=None, lstm_state_critic=None,
                predict_deterministic: bool=False):
        """
        Function of PPO agent which calls the predict function of Brain. Predicts actions given observations.
        Used by the _validation() function of PPO agent and can be accessed from the "outside" to apply on the test set.
        """
        # new observations need to be scaled as well, because they come from the environment "raw"
        new_obs = self.scale_observations(obs=new_obs, env_step_version=env_step_version, n_assets=n_assets)
        # after that, we need to convert the observation to a torch tensor for the network
        new_obs = torch.as_tensor(new_obs, dtype=torch.float)
        new_lstm_obs = torch.as_tensor(new_lstm_obs, dtype=torch.float)
        # without gradient calculation (no backward pass, only forward)
        # (this needs to be specified in pytorch, different than for tensorflow normally)
        with torch.no_grad():
            value, actions, _, _, _, _, \
            lstm_state, lstm_state_actor, lstm_state_critic = self.Brain(observations=new_obs,
                                                                         evaluation_mode=False,# we don't evaluate actions, but predict them
                                                                         actions=None, # hence we don't need to pass any actions to be evaluated'
                                                                         # whether the predictions are deterministic (action means)
                                                                         # or not (sample from distribution)
                                                                         actions_deterministic=predict_deterministic,
                                                                         # only applicable if lstm used, else None
                                                                         lstm_observations=new_lstm_obs,
                                                                         lstm_states=lstm_state,
                                                                         lstm_states_actor=lstm_state_actor,
                                                                         lstm_states_critic=lstm_state_critic)
        # Convert actions and value to numpy array
        actions = actions.detach().numpy()
        value = value.detach().numpy()
        if self.Brain.net_arch == "mlplstm_shared":
            lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
            lstm_state_actor=None
            lstm_state_critic=None
        elif self.Brain.net_arch == "mlplstm_separate":
            lstm_state_actor = (lstm_state_actor[0].detach(), lstm_state_actor[1].detach())
            lstm_state_critic = (lstm_state_critic[0].detach(), lstm_state_critic[1].detach())
            lstm_state=None
        else:
            lstm_state=None
            lstm_state_actor=None
            lstm_state_critic=None

        return actions, value, lstm_state, lstm_state_actor, lstm_state_critic

    def save(self, model_save_path):
        torch.save(self.Brain.state_dict(), model_save_path)

    def scale_observations(self, obs, env_step_version, n_assets):
        # Note depending on the step version we use (see config), we use slightly different inputs
        # and therefore we need to apply some scaling to the observations differently
        # scaling needs to be done before it is used as input data for training or testing,
        # because the inputs are sometimes on totally different scales, like cash for example and asset prices
        # vs macd and volatility

        # Note: cash is a relatively large sum, starting from 1'000'000 in the beginning
        # and as I have observed, it usually goes down over time to something around 20, plus minus
        # The fact that cash is so large can make learning more slow, but we don't want (cannot) to use batch
        # normalization because we initially do forward passes one one single state /observation, which can not
        # be batch-normalized of course. Also, it is practical to have the cash as actual number
        # here in order to calculate the portfolio return etc. and save it to csv for analysis / debugging
        # There are multiple versions to do this, but it makes most sense to use the same scaling factor as
        # for rewards scaling, which is 1e-4
        if env_step_version == "paper":
            # then we have cash at the first place, then asset holdings in the next n_asset places, then asset prices and then other,
            # less problematic features
            obs[0] = obs[0] * 1e-4
            # next come n. asset holdings, which normally vary around 0 and 30000, so lets do 1e-3
            obs[1: n_assets + 1] = obs[1: n_assets + 1] * 1e-3
            # for asset prices, which are the next n_assets number of entries,
            # they are usually between 0+ and 100, we can scale with 1e-2
            # asset prices are also not normalized in the env, because they are practical to use for calculating how
            # many stocks we can buy
            obs[n_assets+1: n_assets+n_assets+1] = obs[n_assets+1: n_assets+n_assets+1] * 1e-2
        elif env_step_version == "newNoShort" or env_step_version == "newNoShort2":
            # in this case, we already have cash weight instead of cash and asset weights instead of asset holdings
            # we only need to scale the asset prices which are there as well
            obs[n_assets+1: n_assets+n_assets+1] = obs[n_assets+1: n_assets+n_assets+1] * 1e-2
        else:
            print("Error: env_step_version not specified correctly.")

        return obs