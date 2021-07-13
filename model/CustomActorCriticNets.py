import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
import logging

# import own libraries
#try:
#    from config.config import *
#except:
#    from config import *

########################################################################
# CUSTOM NETWORK ARCHITECTURES FOR:
#           SHARED FEATURES EXTRACTOR
#           ACTOR NETWORK
#           CRITIC NETWORK
########################################################################

########## SHARED FEATURE EXTRACTOR

class FeatureExtractorNet(nn.Module):
    """
    This is the Shared Feature Extractor Net.
    Input: observations
    Output: intermediate Features
    """

    def __init__(self,
                 observations_size: int = None,
                 mid_features_size: int = 64,
                 hidden_size: int = 64,
                 version: str = "base1"):
        """
        obs_size  : size of the observation space, input dimension
        act_size  : number of actions, will be output dimension of the actor net
        HID_SIZE  : number of neurons
        """
        super(FeatureExtractorNet, self).__init__()
        # define the network which outputs the action_mean
        if version == "base1" or version == "base2":
            # vanilla version with 2 dense layers, each 64 hidden size (neurons), each with a tanh activation
            # and both layers shared between actor and critic. Actor and critic then only
            # have one separate layer each, that s, for the output (actions for actor, value for critic)
            # this is the base case net architecture because it is the same as used in stable baselines
            # as a default and then it is easier to compare if my own ppo implementation is correct or not
            # (performance should be "kind of" in line, although they are using things I did not implement,
            # such as their own custom learning rate scheduling and probably other things I might not have read about,
            # because they do not report all small implementation details which might have an impact))
            self.shared_feature_extractor = nn.Sequential(
                nn.Linear(observations_size, hidden_size, bias=True),
                nn.Tanh(),
                #nn.Softmax(),
                #nn.ReLU(), # relu here makes both layers have 0 gradient always, softmax() as well
                nn.Linear(hidden_size, mid_features_size, bias=True),
                nn.Tanh(),
                #nn.ReLU() # note: when I used Relu activation, the agent gradients were always zero for some reason
            )
            # input: observation_dim
            # output: 64

    def forward(self, observations):
        """
        forward pass through actor network (action_mean_net)
        INPUT: mid_features, the intermediate features we get from the feature extractor
        OUTPUT: the action mean of each action
        """
        mid_features = self.shared_feature_extractor(observations)
        return mid_features


########## ACTOR NETWORK ONLY

class ActorNet(nn.Module):
  """
  This is the Actor Net. It subclasses directly from the pytorch nn.Module class.
  This is a common way to create custom pytorch modules.
  """
  def __init__(self,
               # output dimension = number of actions to predict
               actions_num: int,
               # number of intermediate features (predicted / encoded by the features extractor net)
               mid_features_size: int = 64,
               # hidden size in hidden layer, if there is any
               hidden_size: int = 64,
               version: str = "base1",
               env_step_version: str = "paper",
               ):
    """
    mid_features_size  : size of the observation space / encoded features by feature extractor
    actions_num        : number of actions, will be output dimension of the actor net
    hidden_size        : number of neurons in hidden layer if any
    """
    super(ActorNet, self).__init__()
    # define the network which outputs the action_mean
    if version == "base1":
      # in the base version, as already discussed in the Feature extractor class,
      # the feature extraction is fully shared between actor and critic.
      # actor hence only consists of one additional layer for the actions
      # we want to get actions = how many stocks to buy (:100)
      self.action_mean_net = nn.Sequential(
          nn.Linear(mid_features_size, actions_num, bias=True),
          #nn.Tanh(),
          )
      # in case we are using the "newNoShort" version of the step_version (see documentation),
      # we have to add a softmax layer at the end (because we want to get actions = target portfolio weights)
      if env_step_version == "newNoShort":
          self.action_mean_net = nn.Sequential(
              nn.Linear(mid_features_size, actions_num, bias=True),
              nn.Softmax(),
          )
    # in the base2 version, the network architecture is a bit larger, there is an additional layer for the actor
    # beside the layer for action prediction. Therefore, the hidden size is also reduced a little, in order not to overfit
    elif version == "base2": # an additional layer for the actions net only
        self.action_mean_net = nn.Sequential(
            nn.Linear(mid_features_size, hidden_size-32, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_size-32, actions_num, bias=True),
            #nn.Tanh(), #tanh function as squashing function
        )
        if env_step_version == "newNoShort":
            self.action_mean_net = nn.Sequential(
                nn.Linear(mid_features_size, hidden_size-32, bias=True),
                nn.Tanh(),
                nn.Linear(hidden_size-32, actions_num, bias=True),
                nn.Softmax(dim=0),
            )
    # add log stdev as a parameter, initializes as 0 by default
    # Note: this is because log(std) = 0 means that std = 1, since log(1)=0
    self.log_stdev = nn.Parameter(torch.ones(actions_num)*0)

  def forward(self, mid_features):
    """
    forward pass through actor network (action_mean_net)
    INPUT: mid_features, the intermediate features we get from the feature extractor
    OUTPUT: the action mean of each action
    """
    actions_mean = self.action_mean_net(mid_features)
    return actions_mean


########## CRITIC NETWORK ONLY

class CriticNet(nn.Module):
  """
  This is the Critic Net.
  """
  def __init__(self,
               mid_features_size: int=64,
               hidden_size: int=64,
               version: str="base1"):
    """
    mid_features_size  : size of the observation space, input dimension
    hidden_size        : number of neurons
    OUTPUT             : one value estimate per state, hence output dimension = 1
    """
    super(CriticNet, self).__init__()

    if version == "base1":
      # base1 version is shared feature extractor with actor and critic, and then only
      # one separate layer for critic for value output
      self.value_net = nn.Sequential(#nn.Linear(mid_features_size, hidden_size, bias=True),
                                      #nn.ReLU(),
                                      nn.Linear(mid_features_size, hidden_size, bias=True),
                                      nn.ReLU(),
                                      #nn.Linear(mid_features_size, 1, bias=True),
                                      nn.Linear(hidden_size, 1, bias=True),
                                      )
    elif version == "base2":
          # base2 version adds one more layer for the critic
          self.value_net = nn.Sequential(#nn.Linear(mid_features_size, hidden_size, bias=True),
                                         #nn.ReLU(),
                                         nn.Linear(mid_features_size, hidden_size-32, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size-32, 1, bias=True)
          )


  def forward(self, mid_features):
    """
    Take intermediate features (given by feature extractor) and
    return value estimate for the current state.
    """
    value_estimate = self.value_net(mid_features)
    return value_estimate


########## ACTOR-CRITIC BRAIN (where everything from above comes together

# Helper functions for weights intialization (since it is quite cumbersome if we have a nn.Sequential() model)
# see: https://pytorch.org/docs/stable/nn.init.html
# see: https://discuss.pytorch.org/t/initialising-weights-in-nn-sequential/76553
# about weights initialization, see also:
# https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52
# https://ml-compiled.readthedocs.io/en/latest/initialization.html
# weight initialization is important to overcome the problem of exploding / vanishing gradients in very deep
# neural networks. Since the network here is rather shallow, the way we initialize the weights doesn't have a huge impact.
# (whether we do Xavier or He or orthogonal s/ orthonormal etc.)
def init_weights_feature_extractor_net(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            # fill bias with 0 (if there is a bias, which there always is in my work)
            module.bias.data.fill_(0.0)
def init_weights_actor_net(module, gain=0.01):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            # fill bias with 0 (if there is a bias, which there always is in my work)
            module.bias.data.fill_(0.0)
def init_weights_critic_net(module, gain=1.):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            # fill bias with 0 (if there is a bias, which there always is in my work)
            module.bias.data.fill_(0.0)

### BRAIN CLASS
class BrainActorCritic(nn.Module):
    def __init__(self,
                 observations_size: int,
                 actions_num: int,
                 init_weights_feature_extractor_net = init_weights_feature_extractor_net,
                 init_weights_actor_net = init_weights_actor_net,
                 init_weights_critic_net = init_weights_critic_net,
                 mid_features_size: int = 64,
                 hidden_size_actor: int = 64,
                 hidden_size_critic: int = 64,
                 hidden_size_features_extractor: int = 64,
                 optimizer: torch.optim = torch.optim.Adam,
                 learning_rate: float = 0.001, #0.00025,
                 version: str = "base1",
                 env_step_version: str = "paper",
                 ):
        super(BrainActorCritic, self).__init__()
        self.shared_feature_extractor = FeatureExtractorNet(observations_size=observations_size,
                                                            mid_features_size=mid_features_size,
                                                            hidden_size=hidden_size_features_extractor,
                                                            version=version)
        self.actor = ActorNet(actions_num=actions_num,
                              mid_features_size=mid_features_size,
                              hidden_size=hidden_size_actor,
                              version=version,
                              env_step_version=env_step_version)

        self.critic = CriticNet(mid_features_size=mid_features_size,
                                hidden_size=hidden_size_critic,
                                version=version)

        #initialize orthogonal weights
        self.init_weights_feature_extractor_net = init_weights_feature_extractor_net
        self.init_weights_actor_net = init_weights_actor_net
        self.init_weights_critic_net = init_weights_critic_net

        self.shared_feature_extractor.apply(self.init_weights_feature_extractor_net)
        self.actor.apply(self.init_weights_actor_net)
        self.critic.apply(self.init_weights_critic_net)

        # Setup optimizer with learning rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        # for Adam, apply optimizer to all the parameters in all the three networks
        self.optimizer = self.optimizer(list(self.shared_feature_extractor.parameters()) +
                                        list(self.actor.parameters()) +
                                        list(self.critic.parameters()),
                                        lr=learning_rate, # 0.001 by default
                                        eps=1e-08, # by default
                                        betas=(0.9, 0.999), # by default
                                        weight_decay=0 # L2 penalty, float, 0 by default
                                        )

    def forward(self, observations):
        mid_features = self.shared_feature_extractor(observations)
        actions = self.actor(mid_features)
        value = self.critic(mid_features)
        return actions, value

    def forward_pass(self, observation: torch.Tensor, actions=None, evaluation_mode=False):
        ### FEATURES EXTRACTION
        # get intermediate features from features extractor
        # if features extractor shared between actor and critic, mid_features_actor=mid_features_critic
        mid_features = self.shared_feature_extractor(observation)

        ### CRITIC
        # get value estimate from critic (value network) using these mid_features
        value_estimate = self.critic(mid_features)

        ### ACTOR
        # get estimated action means from actor (policy network) using these mid features
        action_means = self.actor(mid_features)
        #print("agent: action means")
        #print(action_means)
        # get estimated log stdev parameter (like a bias term of the last layer) appended to the actor network
        # Note: it is a nn.Parameter, which is like a tensor added to the module of Pytorch and it is a bit badly
        # documented but apparently this is like a bias term that gets changed as well when the network trains.
        log_stdev = self.actor.log_stdev
        # convert log standard deviation to stdev
        stdev = log_stdev.exp()

        # Note: this is one value. But we need to get a vector of this same value (one for each action)
        # so we can then use it to create a distribution around each action mean
        stdev_vector = torch.ones_like(action_means) * stdev
        # now that we have the means for each action and the standard deviation (one estimate only, same for all actions),
        # we can create a distribution around each action mean; we define a normal distribution for our continuous actions,
        # but we could also use something else
        actions_distribution = Normal(action_means, stdev_vector)
        # see also: https://pytorch.org/docs/stable/distributions.html

        if evaluation_mode:
            # If we are in "evaluation mode", we don't sample a new action but instead
            # use the old action as input to the current distribution.
            # We want to find out: how likely are the actions we have taken during the trajectories sampling
            # (into the Buffer) now, after we have updated our policy with a backward pass?
            # Note: in the first round, we have not yet updated our policy, hence the probabilities we will get will be the same
            # get new action log probabilities for the old actions using peviously defined Normal distribution (actions_distribution)
            actions_log_probs = actions_distribution.log_prob(actions)

            # now we want to calculate the joint distribution of all action mean distributions over all stocks
            # we make the assumption that the actions are independent from each other (could be violated)
            # and we have log probas, hence we can sum the log probabilities up

            # IMPORTANT: if our batch is of length 1, we sum across the first dimension,
            # because we don't want to sum action log probabilities over all days, but just the action log probabilities ove rall actions of ONE day
            # (at first I got a lot of errors because of not considering his and the actor had no gradient)
            # same for entropy below
            if len(actions_log_probs.shape) > 1:
                actions_joint_log_proba = actions_log_probs.sum(dim=1)
            else:
                actions_joint_log_proba = actions_log_probs.sum()

            # calculate joint entropy the same way:
            actions_distr_entropy = actions_distribution.entropy()
            if len(actions_distr_entropy.shape) > 1:
                actions_distr_entropy = actions_distr_entropy.sum(dim=1)
            else:
                actions_distr_entropy = actions_distr_entropy.sum()
            # Note: our log_stdev is initialized as a vector of zeroes.
            # but if we take the exp(0), it returns a vector of ones (the standard deviations).
            # entropy of a Normal distribution with std 1 =~1.4189
            # https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived/1804829
            # so this will be the first entropy value we will get before any backpropagation
            # (this is good to know for debugging / to check if code works properly)
            return value_estimate, actions_joint_log_proba, actions_distr_entropy, action_means, actions_distribution, stdev

        else: # by default,forward pass
            # here we sample actions from the distribution => non-deterministic, in order to add some exploration
            action_samples = actions_distribution.rsample()
            # Note: rsample() supports gradient calculation through the sampler, in contrast to sample()
            # https://forum.pyro.ai/t/sample-vs-rsample/2344

            # get action log probabilities with current action samples.
            # This part of the function is used when we forward pass during trajectories collection into the buffer
            actions_log_probs = actions_distribution.log_prob(action_samples)
            # now we want to calculate the joint distribution of all action mean distributions over all stocks
            # we make the assumption that the actions are independent from each other (could be violated)
            # and we have log probas, hence we can sum the log probabilities up
            if len(actions_log_probs.shape) > 1:
                actions_joint_log_proba = actions_log_probs.sum(dim=1)
            else:
                actions_joint_log_proba = actions_log_probs.sum()
            return value_estimate, action_samples, actions_joint_log_proba, action_means, actions_distribution, stdev

    def predict(self, new_obs, deterministic=False):
        # deterministic / non-deterministic:
        # https://datascience.stackexchange.com/questions/56308/why-do-trained-rl-agents-still-display-stochastic-exploratory-behavior-on-test
        # note: it somehow yields better result in testing when sampling (non-deterministic) actions than when I use the mean directly
        # this is probably because the probability distribution is part of the agent model and therefore model-leaning
        # predictions should be sampled

        # change new observation to torch tensor, if it is not already a torch tensor
        if isinstance(new_obs, torch.Tensor):
            pass
        else:
            new_obs = torch.as_tensor(new_obs, dtype=torch.float)
        # get intermediate features from feature extractor
        mid_features = self.shared_feature_extractor(new_obs)
        # get action means from actor
        action_means = self.actor(mid_features)
        # get value from critic (usually not needed but for debugging)
        value = self.critic(mid_features)
        if deterministic == True:
            # if we predict deterministically, we can just take the action means (most probable action from the distribution)
            predicted_actions = action_means
        elif deterministic == False:
            # if we predict stochastically, we have to sample an action from our distribution,
            # which represents our actual model
            log_stdev = self.actor.log_stdev
            stdev = log_stdev.exp()
            stdev_vector = torch.ones_like(action_means) * stdev
            actions_distribution = Normal(action_means, stdev_vector)
            action_samples = actions_distribution.rsample()
            predicted_actions = action_samples
        predicted_value = value
        # note: prediction here is deterministic; we take the action means (= the output of the actor network)
        # we could also sample an action from our actor network instead (non-deterministic prediction),
        # but if we predict we normally want to use the most likely prediction of our current model for performance evaluation
        return predicted_actions, predicted_value