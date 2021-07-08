import numpy as np
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

# import own libraries
try:
    from config.config import *
except:
    from config import *
from model.CustomOnPolicyBuffer import OnPolicyBuffer
from model.CustomActorCriticNets import FeatureExtractorNet, ActorNet, CriticNet, BrainActorCritic, \
    init_weights_actor_net, init_weights_feature_extractor_net, init_weights_critic_net

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
                 # path where loss curves etc. are saved
                 performance_save_path: str,
                 current_episode: int,
                 # optional validation env for e.g. early stopping based on average reward
                 env_validation=None,
                 # params for neural network
                 batch_size: int=64,
                 num_epochs: int=10,
                 # discount rate for rewards
                 gamma: float=1 / (1 + 0.02),
                 # smoothing value for generalized advantage estimators (exponemtial mean discounting factor)
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
                 ):
        """
        Here, the variables and hyperparameters are initialized.
        """
        self.logger = logger
        ### initialize classes we need for PPO agent construction
        self.Env = env_train
        self.Env_firstday = train_env_firstday
        self.EnvVal = env_validation
        self.EnvVal_firstday = val_env_firstday

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
        # total training steps of experience we wantto collect to the buffer storage
        self.total_timesteps_to_collect = total_timesteps_to_collect

        self.performance_save_path = performance_save_path
        self.current_episode = current_episode

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

        "b" stands for "buffer"
        """
        # get first state / observation from the environment by resetting it
        self.logger.info("enffirstday: "+str(self.Env_firstday))
        obs = self.Env.reset(day=self.Env_firstday, initial=True)
        #self.logger.info("train env reset, first obs: ", obs)
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
        while current_timesteps_collected < total_timesteps_to_collect:
            # FORWARD PASS of current observation based on OLD policy (no updates yet in this module)
            # torch.no_grad() is used to set gradient computation to False. It reduces memory consumption.
            # it can be used here because we will only collect trajectories, and not do backpropagation here,
            # so we don't need the automatical gradient computation
            # see also: https://pytorch.org/docs/stable/generated/torch.no_grad.html
            with torch.no_grad():
                # convert obs (np.array) to torch tensor, so it can be used by the neural network
                obs = torch.as_tensor(obs, dtype=torch.float)
                # get value estimate, sampled action and actions log probabilities from brain,
                # which uses actor and critic architecture
                V_estimate, actions, actions_log_prob, _, _ = self.Brain.forward_pass(observation=obs)
                # Note: this yields a
                # value estimate for the current observation (state) => one value, as tensor of dim (1,)
                # actions for the current obs (state), e.g. one weight for each stock, as tensor of dim (n_assets,)
                # action log probabilities, for the curent obs (state), one for each stock action, as tensor of dim (n_assets,)

            # convert actions to numpy array so it can be used by the environment to get the
            # take a step, get the next state and compute the reward
            # note: must use .detach() on torch.tensors before converting them to numpy because it yields en error otherwise
            # why: because we cannot call numpy on a tensor that requires gradient (requires_grad = True)
            actions = actions.numpy()#.detach().numpy() # .cpu()

            # CLIP ACTIONS
            # this needs to be done because actions are sampled from a Distribution (here: Gaussian,
            # but could also use other, like e.g. Beta distribution), and that leads to actions not
            # necessarily being within the boundaries of the defined action space (note: I am using gym.Box for action space)
            # clipping is used by most implementations online, including stable baselines
            actions_clipped = np.clip(actions, self.Env.action_space.low, self.Env.action_space.high)

            # TAKE A STEP in the environment with the sampled action
            # in order to obtain the new state, a reward and a mask (done; True if
            # end of trajectory (dataset) reached, else False))
            new_obs, reward, done, _ = self.Env.step(actions_clipped)

            # SAVE TRAJECTORY: add obtained experience data to OnPolicyBuffer storage
            self.OnPolicyBuffer.add(obs, "obs", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(actions, "actions", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(actions_clipped, "actions_clipped", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(actions_log_prob, "actions_log_probs", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(V_estimate, "value_estimates", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(reward, "rewards", position=current_timesteps_collected)
            self.OnPolicyBuffer.add(done, "dones", position=current_timesteps_collected)

            # the new observation becomes the "current observation" of the next iteration
            # unless the new observation is the last state; the last state (and the last state value)
            # will not be saved into the buffer, because we don't need them for training later
            # (for example: a forward pass on the last observation in order to obtain new action probabilities to compare with the old
            # action probabilities, is useless, since we have not sampled an action for the last observation. There is no step coming after
            # the last observation.)
            obs = new_obs
            # collect trajectories step by step until total_rb_timesteps are reached
            current_timesteps_collected += 1

            # if done = True, we have reached the end of our data set and we break the
            # loop. This applies if total_timesteps_to_collect have been set higher than the actually
            # available time steps in the data set
            if current_timesteps_collected in list(range(0, total_timesteps_to_collect, 1000)) + \
                    [total_timesteps_to_collect]:
                self.logger.info(f"current timesteps collected: {current_timesteps_collected + 1} / max. {total_timesteps_to_collect}")
                # self.logger.info("\nactions before clipping: ")
                # self.logger.info(actions)
                # self.logger.info("\nactions after clipping: ")
                # self.logger.info(actions_clipped)
                # self.logger.info(f"\naction log probs: ")
                # self.logger.info(actions_log_prob)
                # self.logger.info(f"\nvalue estimate: {V_estimate}")
                #self.logger.info(f"\nreward : {reward}")
                #self.logger.info("saved old log probs: ")
                #self.logger.info(actions_log_prob)
            if done:
                self.logger.info("experience collection finished (because episode finished  (done)). ")
                break

        # now we need to get the value estimates for the terminal state, the new_obs
        # we need the terminal value estimate in order tp compute the advantages below
        with torch.no_grad():
            obs = torch.as_tensor(new_obs, dtype=torch.float)
            V_terminal_estimate, _, _, _, _ = self.Brain.forward_pass(observation=obs)

        self.OnPolicyBuffer.calculate_and_store_advantages(terminal_V_estimate=V_terminal_estimate,
                                                           gamma=self.gamma,
                                                           gae_lambda=self.gae_lambda)
        self.OnPolicyBuffer.calculate_and_store_returns()

    def learn(self,
              total_timesteps: int=100000,
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
        # we do this until we have learned for total_learning_timesteps

        # for saving later
        suffix = "epoch_"
        epoch_colnames = [suffix + str(s) for s in list(range(1, self.num_epochs + 1))]

        # at the end of every 10 epochs, validation rewards are calculated on the holdout data set
        validation_rewards = []

        while learning_timesteps_done < total_learning_timesteps:
            self.logger.info(f"\n---TRAINING_TIMESTEPS_DONE: {learning_timesteps_done} / {total_learning_timesteps}")

            # collect experience in the environment based on the current policy and store in buffer
            self._collect_experiences_to_buffer(total_timesteps_to_collect=self.total_timesteps_to_collect)

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

            # now we train for multiple epochs
            for epoch in range(1, self.num_epochs + 1):
                self.logger.info(f"---EPOCH: {epoch} / {self.num_epochs}")

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
                    start_idx += self.batch_size

                    self.logger.info(f"BATCH NUMBER: {batch_num + 1}")
                    if epoch == 1 and start_idx == 0:
                        self.logger.info(f"sample batch (observations), (len: {len(batch_obs)})")
                        self.logger.info(batch_obs)

                    #if batch_num == 1:
                        #self.logger.info("getting old log probs as batch: ")
                        #self.logger.info(batch_actions_log_probs)
                    #self.logger.info("batch_obs")
                    #self.logger.info(batch_obs)
                    #self.logger.info("batch_actions")
                    #self.logger.info(batch_actions)
                    #self.logger.info("batch log probs")
                    #self.logger.info(batch_actions_log_probs)
                    #self.logger.info("batch_returns")
                    #self.logger.info(batch_returns)
                    #self.logger.info("batch_advantages")
                    #self.logger.info(batch_advantages)

                    # Standardizing advantage estimates (creating z-score) across each batch
                    # (not in the paper)
                    # many implementations use a normalized version because it makes convergence
                    # more stable and also decreases the variance of our advantage estimates
                    # many people say that it therefore makes learning more stable
                    # at the end, a very small number s added (e.g. 1e-100) so that we never divide by 0
                    advantage_est_standardized = (batch_advantages - batch_advantages.mean()) / \
                                                 (batch_advantages.std() + 1e-8)

                    # convert the actions from the batch (old actions, since these are actions with the "old" policy, the one
                    # we used to sample trajectories into the buffer) to torch tensor, since we are going to use them so
                    # sample the new action log probabilities
                    old_actions = batch_actions #torch.as_tensor(batch_actions, dtype=torch.float)
                    # Get value estimates, current action log probabilities and action distribution entropy
                    # with current weights from critic doing a forward pass
                    # NOTE: in the first round the forward pass will yield the same actions / action distr. / action log probs.
                    # as when we collected trajectories into the OnPolicyBuffer, hence the probabilities ratio (calculated later) will be 1
                    # NOTE: this time we should not do the forward pass with "with torch.no_grad()", because we will later call backward()
                    # and want to be able to compute the gradients in order to update our policy
                    new_V_estimate, current_action_new_log_prob, action_distr_entropy, _, _ = \
                        self.Brain.forward_pass(observation=torch.as_tensor(batch_obs, dtype=torch.float),
                                                actions=old_actions,
                                                evaluation_mode=True)  # evaluation mode must be true: see documentation in Brain class
                    # Calculate the probabilities ratio (since we have log probabilities, we can simply
                    # subtract them in the exponential function and we get back the "original", non-log probabilities)
                    # log probabilities are used as a convention, because they make calculations easier
                    # see also: https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                    # note: the log probabilities are scalars, hence the probability ratio will be a scalar value as well, not a vector of ratios
                    # self.logger.info("current action log probs: ")
                    # self.logger.info(current_action_new_log_prob)
                    # self.logger.info("action log probs: " )
                    # self.logger.info(batch_actions_log_probs)
                    #self.logger.info("action new log prob:")
                    #self.logger.info(current_action_new_log_prob)
                    #self.logger.info("action old log prob:")
                    #self.logger.info(batch_actions_log_probs)

                    proba_ratio = torch.exp(current_action_new_log_prob - batch_actions_log_probs)
                    #self.logger.info("proba ration: ")
                    #self.logger.info(proba_ratio)
                    # in the first iteration (before first policy update),
                    # the ratio should be 1, since we will sample the same actions with the same net weights every time
                    ###  ACTOR LOSS (POLICY LOSS, SURROGATE LOSS)
                    # r * A
                    surr_loss_1 = proba_ratio * advantage_est_standardized
                    #self.logger.info("surr loss 1: ", surr_loss_1)
                    #self.logger.info(surr_loss_1)

                    # clipped_r * A
                    surr_loss_2 = torch.clamp(proba_ratio,
                                              min=1 - self.clip_epsilon,
                                              max=1 + self.clip_epsilon) * advantage_est_standardized
                    #self.logger.info("surr loss 2: ")
                    #self.logger.info(surr_loss_2)

                    # Note: because we will use gradient descent, not gradient Ascent, we need to take the negative of the surrogate function
                    # (in the paper, they maximize the (non-negative) surrogate function / loss with Gradient ascent)
                    surr_loss = torch.min(surr_loss_1, surr_loss_2)
                    #self.logger.info("surr loss: ")
                    #self.logger.info(surr_loss)

                    # calculate the clipped surrogate los (=actor loss / policy network loss)
                    # since the actor loss in the paper is actually defined as "actor gain", so we would have to maximize it with
                    # gradient ascent; but here we want to use gradient descent so we will take the negative of the surrogate loss mean
                    actor_loss = torch.mean(-surr_loss)
                    #self.logger.info("actor loss: ")
                    #self.logger.info(actor_loss)

                    ### CRITIC LOSS (VALUE LOSS)
                    # Value loss using the TD(gae_lambda) target in the calculation of the returns (hence more smooth, less variance)
                    # it is the mean squared error between the estimated state value V and the actual returns (smoothed)
                    # self.logger.info("new v est: ")
                    # self.logger.info(new_V_estimate)
                    # self.logger.info("batch returns: ")
                    # self.logger.info(batch_returns)
                    # we need to flatten the new value estimates, because its a tensor like this: [[v1],[v1], ...]
                    # and we need it to be like this: [v1,v2,...] (rewards are also like this), else we et an error below
                    new_V_estimate = new_V_estimate.flatten()
                    # Note: since critic loss = MSE loss (mean squared error), it is always going to be positive
                    critic_loss = nn.functional.mse_loss(new_V_estimate, batch_returns)  # returns used as target value
                    #self.logger.info("critic loss: ")
                    #self.logger.info(critic_loss)

                    ### ENTROPY LOSS
                    entropy_loss = -torch.mean(action_distr_entropy)
                    #self.logger.info("entropy loss: ")
                    #self.logger.info(entropy_loss)

                    # TOTAL LOSS FUNCTION:
                    # Note: the total loss for gradient ascent would be: actor_loss - c1*critic_loss + c2*entropy_loss,
                    # but since we do gradient descent, actor_loss and entropy loss are negative, value loss is positive (opposite signs)
                    total_loss = actor_loss + self.critic_loss_coef * critic_loss + self.entropy_loss_coef * entropy_loss
                    #self.logger.info("total loss:")
                    #self.logger.info(total_loss)

                    # UPDATING THE ACTOR-CRITIC MODEL (Updates policy, feature extractor and value network)
                    # Note: we first need to call zero_grad() on the optimizer, in order to clear all the gradients from the previous iteration,
                    # because otherwise we would be accumulating gradients over multiple iterations (batches) for the update, which would not be good
                    # then we use loss.backward() to backpropagate the loss and compute the current gradients (derivatives w.r.t. the parameters) for the current batch
                    # then we use a gradient clipping method in order to make sure that our gradients don't "explode"
                    # see also: https://www.reddit.com/r/MachineLearning/comments/4qshk3/max_norm_gradientweight_clipping_for/
                    # then we use optimizer.step(), with this we take a step in the direction of the graidnet: this is the actual update step, where our weights are updated
                    # see also: https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301
                    self.Brain.optimizer.zero_grad()
                    total_loss.backward()
                    # avoiding exploding gradients with gradient normalization:
                    # https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/
                    # default value the same as the one from stable baselines for reproducibility purposes
                    torch.nn.utils.clip_grad_norm_(self.Brain.parameters(), self.max_gradient_normalization)
                    self.Brain.optimizer.step()

                    # AT THE END OF EACH BATCH:
                    # append metrics
                    # metrics that come as a multi-dim tensor per batch (one entry per observation in batch)
                    # each of length batch_size (e.g. 64)
                    # => list of lists => flatten list so that we get only a list of values, one for each obs
                    prob_ratio_of_epoch.append(np.array(proba_ratio.detach()).flatten())
                    surr_loss_1_of_epoch.append(np.array(surr_loss_1.detach()).flatten())
                    surr_loss_2_of_epoch.append(np.array(surr_loss_2.detach()).flatten())
                    surr_loss_of_epoch.append(np.array(surr_loss.detach()).flatten())
                    advantages_of_epoch.append(np.array(advantage_est_standardized.detach()).flatten())
                    # metrics that come as single values per batch (each of length 1)
                    # => list of values
                    actor_loss_of_epoch.append(actor_loss.detach())
                    critic_loss_of_epoch.append(critic_loss.detach())
                    entropy_loss_of_epoch.append(entropy_loss.detach())
                    combined_loss_of_epoch.append(total_loss.detach())

                    #update learning timesteps only for first (or any, just only one) epoch,
                    # because we only want to count the data samples we train on (not how often we trained on them)
                    if epoch == 1:
                        self.logger.info(f"learning timesteps before update (total to do: {total_learning_timesteps}):")
                        self.logger.info(learning_timesteps_done)
                        learning_timesteps_done += self.batch_size
                        #if learning_timesteps_done >= total_learning_timesteps:
                            #self.logger.info(f"learning timesteps reached: {learning_timesteps_done}  / "
                                        # f"total {total_learning_timesteps}."
                                        # f"\nTRAINING ROUND BREAK.")
                            #break
                        self.logger.info("learning timesteps after update:")
                        self.logger.info(learning_timesteps_done)
                    #for name, param in self.Brain.named_parameters():
                        #self.logger.info(f"name    : {name}, \n{param}")
                        #self.logger.info(f"gradient: {param.grad}")

                # AT THE END OF ALL BATCHES (ONE EPOCH)
                #self.logger.info(f"Avg loss per epoch: {np.mean(epoch_loss)}")
                # after each epoch end, append
                # (each below will then be a list of lists)
                # => list (for all epochs together) of lists (one for each epoch) of values (one for each obs in batch, e.g. 64)
                # [[v,v,v],[v,v,v],[v,v,v],...] = [epoch1, epoch2, ...]
                advantages_all_epochs.append(np.array(advantages_of_epoch).flatten())
                prob_ratio_all_epochs.append(np.array(prob_ratio_of_epoch).flatten())
                surr_loss_1_all_epochs.append(np.array(surr_loss_1_of_epoch).flatten())
                surr_loss_2_all_epochs.append(np.array(surr_loss_2_of_epoch).flatten())
                surr_loss_all_epochs.append(np.array(surr_loss_of_epoch).flatten())
                actor_loss_all_epochs.append(actor_loss_of_epoch)
                critic_loss_all_epochs.append(critic_loss_of_epoch)
                entropy_loss_all_epochs.append(entropy_loss_of_epoch)
                combined_loss_all_epochs.append(combined_loss_of_epoch)

                self.logger.info(f"---EPOCH: {epoch} / {self.num_epochs} done.")

            # AT THE END OF ALL EPOCHS
            # after all 10 (or other) epochs, we save the data to csv
            for li, liname in zip([advantages_all_epochs, prob_ratio_all_epochs, surr_loss_1_all_epochs,
                                   surr_loss_2_all_epochs, surr_loss_all_epochs],
                                  ["advantages_all_epochs", "prob_ratio_all_epochs", "surr_loss_1_all_epochs",
                                   "surr_loss_2_all_epochs", "surr_loss_all_epochs"]):
                pd.DataFrame(np.transpose(li),
                         columns=epoch_colnames).to_csv(os.path.join(self.performance_save_path,
                                                                         f"{liname}_"
                                                                         f"ep{self.current_episode}_"
                                                                         "LearningTimestepsDone_"
                                                                         f"{learning_timesteps_done}.csv"))
            pd.DataFrame({"actor_loss": np.array(actor_loss_all_epochs).flatten(),
                          "critic_loss": np.array(critic_loss_all_epochs).flatten(),
                          "entropy_loss": np.array(entropy_loss_all_epochs).flatten(),
                          "combined_loss": np.array(combined_loss_all_epochs).flatten(),
                          }).to_csv(os.path.join(self.performance_save_path,
                                                 f"train_performances_"
                                                 f"ep{self.current_episode}_"
                                                 f"LearningTimestepsDone_{learning_timesteps_done}.csv"))

            self.logger.info(f"Validation beginning.")
            # after all 10 epochs, if we have passed a validation env, we do some "out of sample testing"
            if self.EnvVal is not None:
                self._validation()
            self.logger.info(f"Validation ended.")

    def _validation(self) -> None:
        validation_rewards = []
        obs_val = self.EnvVal.reset(day=self.EnvVal_firstday)
        for j in range(len(self.EnvVal.df.index.unique())):
            val_actions, val_value = self.predict(obs_val)
            obs_val, val_rewards, val_done, _ = self.EnvVal.step(val_actions)
            validation_rewards.append(val_rewards)
            if val_done:
                break
        self.logger.info(f"validation mean reward: {np.mean(validation_rewards)}")
        return None # Note: rewards are already saved to csv by the validation env


    def predict(self, new_obs):
        new_obs = torch.as_tensor(new_obs, dtype=torch.float)
        # without gradient calculation (no backward pass, only forward)
        # (this needs to be specified in pytorch, different than for tensorflow normally)
        with torch.no_grad():
            actions, value = self.Brain.predict(new_obs)
        # Convert actions and value to numpy array
        actions = actions.detach().numpy()
        value = value.detach().numpy()
        return actions, value

    def save(self, model_save_path):
        torch.save(self.Brain.state_dict(), model_save_path)


