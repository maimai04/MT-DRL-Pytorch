import torch
from torch import nn
from torch.distributions import Normal
import numpy as np


########################################################################
# CUSTOM NETWORK ARCHITECTURES FOR:
#           FEATURES EXTRACTOR
#           ACTOR NETWORK
#           CRITIC NETWORK
#           BRAIN NETWORK: Where all of the above is combined
########################################################################

########## SHARED FEATURE EXTRACTOR

class FeatureExtractorNet(nn.Module):
    """
    This is the Shared Feature Extractor Net.
    Input: observations
    Output: intermediate Features

    Note: MLP = Multilayer Perceptron
          LSTM = Long short-term memory
    """
    def __init__(self,
                 observations_size: int = None, # usually n_assets * n_asset_features + 1 (cash) + 1 (vix)
                 mid_features_size: int = 64, # number of neurons in the mlp layer
                 hidden_size: int = 64,
                 version: str = "mlp_separate",
                 lstm_observations_size: int = None,  # usually n_assets (returns) + 1 (vix)
                 lstm_hidden_size: int = 64, # number of neurons in the lstm layer
                 lstm_num_layers: int = 2, # how many lstm layers; is a parameter one can directly pass to the nn.LSTM
                 ):
        """
        obs_size  : size of the observation space, input dimension
        act_size  : number of actions, will be output dimension of the actor net
        HID_SIZE  : number of neurons
        """
        super(FeatureExtractorNet, self).__init__()
        self.observations_size = observations_size
        self.mid_features_size = mid_features_size
        self.hidden_size = hidden_size
        self.version = version
        self.lstm_observations_size = lstm_observations_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # define the network which outputs the action_mean
        if version == "mlp_separate" or version == "mlp_shared":
            print("(FE) version = only mlp (shared or separate)")
            # mlp_separate: base version according to paper implementation, NO shared layers between actor and critic
            # mlp_shared: base version but with shared layers
                # vanilla version with 2 dense layers, each 64 hidden size (neurons), each with a tanh activation
                # and both layers shared between actor and critic. Actor and critic then only
                # have one separate layer each, that is, for the output (actions for actor, value for critic)
                # this is the base case net architecture because it is the same as used in stable baselines
                # as a default and then it is easier to compare if my own ppo implementation is correct or not
                # (performance should be "kind of" in line, although they are using things I did not implement,
                # such as their own custom learning rate scheduling and probably other things I might not have read about,
                # because they do not report all small implementation details which might have an impact))
            self.feature_extractor = nn.Sequential(
                nn.Linear(in_features=observations_size, out_features=hidden_size, bias=True),
                nn.Tanh(),
                #nn.Softmax(),
                #nn.ReLU(), # relu and softmax here didn't work well; made both layers have 0 gradient more often
                nn.Linear(in_features=hidden_size, out_features=mid_features_size, bias=True),
                nn.Tanh(),
            )

        elif version == "mlplstm_separate" or version == "mlplstm_shared": # using mlp as before and adding an lstm for additional lags extraction.
            print("(FE) version = mlp + lstm")
            # again, no shared layers between actor and critic
            self.feature_extractor_mlp = nn.Sequential(
                nn.Linear(in_features=observations_size, out_features=hidden_size, bias=True),
                nn.Tanh(),
                nn.Linear(in_features=hidden_size, out_features=mid_features_size, bias=True),
                nn.Tanh(),
            )
            # add LSTM module
            # lstm in pytorch: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            # see also tutorial: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
            # input_size = number of features, hidden_size = number of features in hidden state h, num_layers = number of LSTM layers (stacked LSTM if >1)
            # biase = True, by default, batch_first = True (False by default);
            # means that input is provided as (batch_size, sequence_length, n_features(=input_size)
            #note: here is the source code of nn.LSTM (pytorch) https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
            # for understanding how the forward method works on this nn.Module
            self.feature_extractor_lstm = nn.LSTM(input_size=lstm_observations_size, hidden_size=lstm_hidden_size,
                                                         num_layers=lstm_num_layers, batch_first=True, bias=True)
            # use a summary layer to bring down the feature space to 64 (since now we have two outputs),
            # so that the output can be then directly used in the actor resp. critic network. Using Tanh activation
            self.summary_layer = nn.Sequential(
                nn.Linear(in_features=mid_features_size+lstm_hidden_size, out_features=mid_features_size),
                nn.Tanh()
            )
        else:
            print("(FE) Error, no valid version specified.")

    def create_initial_lstm_state(self, sequence_length=1):
        # initialize zero hidden states i the beginning
        # see: https://discuss.pytorch.org/t/initialization-of-first-hidden-state-in-lstm-and-truncated-bptt/58384
        # shape of hidden state h / cell state c is each: (num_layers, number_timesteps, hidden_size)
        # e.g. if we have 2 layers (default), a batch of 5 timesteps and hidden_size=64 (default) => shape= (2, 5, 64)
        # when we do a forward pass or predict on one sample (day) only, we have (2,1,64)
        initial_lstm_states = (torch.zeros(self.lstm_num_layers, sequence_length, self.lstm_hidden_size),
                               torch.zeros(self.lstm_num_layers, sequence_length, self.lstm_hidden_size))
        return initial_lstm_states

    def forward(self, observations: torch.Tensor=None, lstm_observations: torch.Tensor=None, lstm_states=None):
        """
        forward pass through actor network (action_mean_net)
        INPUT: mid_features, the intermediate features we get from the feature extractor
        OUTPUT: the action mean of each action
        """
        # if length of shape vector is 1 (e.g. shape = 2), then we have a tensor like this: [features],
        # => batch of 1
        # and we want to make it compatible with concatenation later with the lstm output, so we reshape it to [[features]] (shape = batch_size, n_features)
        if len(observations.shape) == 1 and self.version in ["mlplstm_separate", "mlplstm_shared"]:
            #print("observations in (FE): ")
            #print(observations.shape)
            observations = observations.reshape(1, self.observations_size)
            #print("observations out (FE): ")
            #print(observations)
            #print(observations.shape)
            #print("lstm_observations in (FE):")
            #print(lstm_observations.shape)
            # reshape the lstm input dimension_ must be 3D (batch_number=1, time steps per batch = 1, features_number)
            lstm_observations = lstm_observations.reshape(1, 1, self.lstm_observations_size)
            #print("lstm_observations out (FE):")
            #print(lstm_observations.shape)

        elif len(observations.shape) > 1 and self.version in ["mlplstm_separate", "mlplstm_shared"]:
            #print("lstm_observations in (FE):")
            #print(lstm_observations.shape)
            # reshape the lstm input dimension_ must be 3D (batch_number=1, time steps per batch, features_number)
            lstm_observations = lstm_observations.reshape(1, len(observations), self.lstm_observations_size)
            #print("lstm_observations out (FE):")
            #print(lstm_observations.shape)

        # if we have only mlp, we use simply the mlp feature extractor (actually same as feature_extractor_mlp)
        if self.version == "mlp_separate" or self.version == "mlp_shared":
            mid_features = self.feature_extractor(observations)
        # if we have an lstm additionally:
        elif self.version == "mlplstm_separate" or self.version == "mlplstm_shared":
            mid_features_mlp = self.feature_extractor_mlp(input=observations)
            # observations for the lstm layer need to be of dim 3, even if it is just one observation for one forward pass
            # it must be (batchsize, timesteps, features). If we pass only ONE TIME STEP, shape must be: torch.Size([1, 1, lstm_observations_shape])
            # Note: lstm_states = (h, c) with h= hidden state, c = cell state
            #if len(observations) ==64:
                #print("lstm input dim:")
                #print(lstm_observations.dim())
            mid_features_lstm, lstm_states = self.feature_extractor_lstm(input=lstm_observations, hx=lstm_states)
            # reshape the features received from the mlp feature extractor
            mid_features_mlp_reshaped = mid_features_mlp.reshape(len(observations),  self.mid_features_size)
            #if len(observations) ==64:
                #print("mid_features mlp reshaped: ")
            #    print(mid_features_mlp_reshaped.shape)
            #    print(mid_features_mlp_reshaped)
            # reshape the features received from the lstm feature extractor
            mid_features_lstm_reshaped = mid_features_lstm.reshape(len(observations),  self.mid_features_size)
            #if len(observations) ==64:
                #print("mid_features lstm: ")
                #print(mid_features_lstm.shape)
                #print(mid_features_lstm)
            #    print("mid_features lstm reshaped: ")
            #    print(mid_features_lstm_reshaped.shape)
                #print(mid_features_lstm_reshaped)
            # concatenate features on dim 1
            mid_features = torch.cat((mid_features_mlp_reshaped, mid_features_lstm_reshaped), dim=1)
            #if len(observations) ==64:
            #    print("mid_features combined: ")
            #    print(mid_features.shape)
                #print(mid_features)
                #print(mid_features[0])

            # simple linear in order to bring together the two features down to a smaller size
            mid_features = self.summary_layer(mid_features)

        else:
            print("(FE) Error, no valid version specified.")
        return mid_features, lstm_states


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
               version: str = "mlp_separate",
               env_step_version: str = "paper",
               ):
    """
    mid_features_size  : size of the observation space / encoded features by feature extractor
    actions_num        : number of actions, will be output dimension of the actor net
    hidden_size        : number of neurons in hidden layer if any
    """
    super(ActorNet, self).__init__()
    # define the network which outputs the action_mean
    #if version == "mlp_separate" or version == "mlp_shared": # todo: rm, independent of net version
      # in the base version, as already discussed in the Feature extractor class,
      # the feature extraction is fully shared between actor and critic.
      # actor hence only consists of one additional layer for the actions
      # we want to get actions = how many stocks to buy (:100)
    self.action_mean_net = nn.Sequential(
        nn.Linear(in_features=mid_features_size, out_features=actions_num, bias=True),
        #nn.Tanh(), # I first used the squashing function but git better results without it, for some reason
        )
    # in case we are using the "newNoShort" version of the step_version (see documentation),
    # we have to add a softmax layer at the end (because we want to get actions = target portfolio weights)
    if env_step_version == "newNoShort":
        self.action_mean_net = nn.Sequential(
          nn.Linear(in_features=mid_features_size, out_features=actions_num, bias=True),
          nn.Softmax(),
      )
    # add log stdev as a parameter, initializes as 0 by default
    # Note: this is because log(std) = 0 means that std = 1, since log(1)=0

    # note: sometimes problems that stdev EXPLODES; based on a reddit post (https://www.reddit.com/r/reinforcementlearning/comments/fdzbs9/ppo_entropy_and_gaussian_standard_deviation/)
    # i found a workaround in the openai spinning up implementation: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L84
    log_stdev = -0.5 * np.ones(actions_num, dtype=np.float32)
    self.log_stdev = torch.nn.Parameter(data=torch.as_tensor(log_stdev), requires_grad=True)
    #self.log_stdev = nn.Parameter(torch.ones(actions_num)*0)

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
               version: str="mlp_separate"):
    """
    mid_features_size  : size of the observation space, input dimension
    hidden_size        : number of neurons
    OUTPUT             : one value estimate per state, hence output dimension = 1
    """
    super(CriticNet, self).__init__()
    # tried out multiple architectures but went for the current one because it is the one from the PPO paper and
    # also the one from the ensemble paper, so easier to compare performances
    self.value_net = nn.Sequential(#nn.Linear(mid_features_size, hidden_size, bias=True),
                                  #nn.ReLU(),
                                  #nn.Linear(mid_features_size, hidden_size, bias=True),
                                  #nn.ReLU(),
                                  #nn.Tanh()
                                  nn.Linear(in_features=mid_features_size, out_features=1, bias=True),
                                  #nn.Linear(hidden_size, 1, bias=True),
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
# what to use?
# https://jamesmccaffrey.wordpress.com/2020/11/20/the-gain-parameter-for-the-pytorch-xavier_uniform_-and-xavier_normal_-initialization-functions/
def init_weights_feature_extractor_net(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        #nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            # fill bias with 0 (if there is a bias, which there always is in my work)
            module.bias.data.fill_(0.0)
    #if isinstance(module, nn.LSTM):
    #    nn.init.xavier_normal_(module.weight, gain=gain)
    #    if module.bias is not None:
            # fill bias with 0 (if there is a bias, which there always is in my work)
    #        module.bias.data.fill_(0.0)
def init_weights_actor_net(module, gain=0.01):
    if isinstance(module, nn.Linear):
        #nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            # fill bias with 0 (if there is a bias, which there always is in my work)
            module.bias.data.fill_(0.0)
def init_weights_critic_net(module, gain=1.):
    if isinstance(module, nn.Linear):
        #nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.xavier_normal_(module.weight, gain=gain)
        if module.bias is not None:
            # fill bias with 0 (if there is a bias, which there always is in my work)
            module.bias.data.fill_(0.0)

### BRAIN CLASS
class BrainActorCritic(nn.Module):
    def __init__(self,
                 observations_size: int,
                 actions_num: int,
                 feature_extractor_class=FeatureExtractorNet,
                 actor_class=ActorNet,
                 critic_class=CriticNet,
                 init_weights_feature_extractor_net = init_weights_feature_extractor_net,
                 init_weights_actor_net = init_weights_actor_net,
                 init_weights_critic_net = init_weights_critic_net,
                 mid_features_size: int = 64,
                 hidden_size_actor: int = 64,
                 hidden_size_critic: int = 64,
                 hidden_size_features_extractor: int = 64,
                 # for lstm, only actually used if verison = mlplstm
                 lstm_observations_size: int = None,
                 lstm_hidden_size_feature_extractor: int = 64,
                 lstm_num_layers: int = 2, #default is 2 because that worked well
                 # optimizer
                 optimizer: torch.optim = torch.optim.Adam,
                 learning_rate: float = 0.00025,
                 # versions
                 version: str = "mlp_separate",
                 env_step_version: str = "paper",
                 ):
        """

        @param observations_size:
        @param actions_num:
        @param feature_extractor_class:
        @param actor_class:
        @param critic_class:
        @param init_weights_feature_extractor_net:
        @param init_weights_actor_net:
        @param init_weights_critic_net:
        @param mid_features_size:
        @param hidden_size_actor:
        @param hidden_size_critic:
        @param hidden_size_features_extractor:
        @param lstm_observations_size:
        @param lstm_hidden_size_feature_extractor:
        @param lstm_num_layers:
        @param optimizer:
        @param learning_rate:
        @param version:
        @param env_step_version:
        """
        super(BrainActorCritic, self).__init__()
        # initialize class variables (only those needed in other functions)
        self.version = version

        if version == "mlp_shared" or version == "mlplstm_shared":
            # only one feature extractor for both actor and critic (shared)
            self.feature_extractor = feature_extractor_class(observations_size=observations_size,#FeatureExtractorNet
                                                         mid_features_size=mid_features_size,
                                                         hidden_size=hidden_size_features_extractor,
                                                         version=version,
                                                         # if we have version == mlplstm, else these will be ignored
                                                         lstm_hidden_size=lstm_hidden_size_feature_extractor,
                                                         lstm_observations_size=lstm_observations_size,
                                                         lstm_num_layers=lstm_num_layers)
            # initialize weights for feature extractor
            self.init_weights_feature_extractor_net = init_weights_feature_extractor_net
            self.feature_extractor.apply(self.init_weights_feature_extractor_net)
        elif version == "mlp_separate" or version == "mlplstm_separate":
            # separate feature extraction layers between actor and critic
            self.feature_extractor_actor = FeatureExtractorNet(observations_size=observations_size,
                                                               mid_features_size=mid_features_size,
                                                               hidden_size=hidden_size_features_extractor,
                                                               version=version,
                                                               # if we have verison == mlplstm, else these will be ignored
                                                               lstm_hidden_size=lstm_hidden_size_feature_extractor,
                                                               lstm_observations_size=lstm_observations_size,
                                                               lstm_num_layers=lstm_num_layers
                                                               )
            self.feature_extractor_critic = FeatureExtractorNet(observations_size=observations_size,
                                                               mid_features_size=mid_features_size,
                                                               hidden_size=hidden_size_features_extractor,
                                                               version=version,
                                                               # if we have verison == mlplstm, else these willmbe ignored
                                                               lstm_hidden_size=lstm_hidden_size_feature_extractor,
                                                               lstm_observations_size=lstm_observations_size,
                                                               lstm_num_layers=lstm_num_layers
                                                               )
            # initialize weights for both feature extractor nets separately
            self.init_weights_feature_extractor_net = init_weights_feature_extractor_net
            self.feature_extractor_actor.apply(self.init_weights_feature_extractor_net)
            self.feature_extractor_critic.apply(self.init_weights_feature_extractor_net)
        else:
            print("(BRAIN) Error, no valid version specified.")

        # initialize actor net
        self.actor = actor_class(actions_num=actions_num,
                              mid_features_size=mid_features_size,
                              hidden_size=hidden_size_actor,
                              version=version,
                              env_step_version=env_step_version)

        # initialize acritic net
        self.critic = critic_class(mid_features_size=mid_features_size,
                                hidden_size=hidden_size_critic,
                                version=version)

        #initialize weights for actor and critic
        self.init_weights_actor_net = init_weights_actor_net
        self.init_weights_critic_net = init_weights_critic_net
        self.actor.apply(self.init_weights_actor_net)
        self.critic.apply(self.init_weights_critic_net)

        # Setup optimizer with learning rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        if version == "mlp_shared" or version == "mlplstm_shared":
            # for Adam, apply optimizer to all the parameters in all the three networks
            self.optimizer = self.optimizer(list(self.feature_extractor.parameters()) +
                                            list(self.actor.parameters()) +
                                            list(self.critic.parameters()),
                                            lr=learning_rate, # 0.00025 by default
                                            eps=1e-08, # by default
                                            betas=(0.9, 0.999), # by default
                                            weight_decay=0 # L2 penalty, float, 0 by default
                                            )
        elif version == "mlp_separate" or version == "mlplstm_separate": # separate layers between actor and critic
            self.optimizer = self.optimizer(list(self.feature_extractor_actor.parameters()) +
                                            list(self.feature_extractor_critic.parameters()) +
                                            list(self.actor.parameters()) +
                                            list(self.critic.parameters()),
                                            lr=learning_rate, # 0.00025 by default
                                            eps=1e-08, # by default
                                            betas=(0.9, 0.999), # by default
                                            weight_decay=0 # L2 penalty, float, 0 by default
                                            )
        else:
            print("(BRAIN) Error, no valid version specified.")

    def forward(self, observations, lstm_observations=None, lstm_states=None, lstm_states_actor=None, lstm_states_critic=None):
        """
        Note: this function must be implemented when subclassing nn.Module, however it is not actually called during my training,
        because I use a custom forward method: forward_call (see next function).
        @param observations: Input observations one state or batch of states, coming from the environment, should be torch.Tensor
        @param lstm_observations: Input observations only for the lstm. Only applicable if lstm_shared or lstm_separate net version specified. By default None.
        @param lstm_states: hidden and cell state tuple from forward pass from the previous state. encodes previous part of time series
                            only applicable if lstm net version SHARED chosen (so same lstm state for actor and critic, since shared feature extractor)
        @param lstm_states_actor: hidden and cell state tuple from forward pass from the previous state for the ACTOR. encodes previous part of time series
                            only applicable if lstm net version SEPARATE chosen
        @param lstm_states_critic: hidden and cell state tuple from forward pass from the previous state for the CRITIC. encodes previous part of time series
                            only applicable if lstm net version SEPARATE chosen
        @return:
        """
        if self.version == "mlp_shared" or self.version == "mlplstm_shared": # shared feature extactor between actor and critic
            mid_features, lstm_state = self.feature_extractor(observations=observations,
                                                              lstm_observations=lstm_observations,
                                                              lstm_states=lstm_states)
            actions = self.actor(mid_features)
            value = self.critic(mid_features)

            # set variables not needed
            lstm_states_actor = None
            lstm_states_critic = None
            if self.version == "mlp_shared":
                lstm_states=None

        elif self.version == "mlp_separate" or self.version == "mlplstm_separate": # separate layers between actor and critic
            mid_features_actor, lstm_states_actor = self.feature_extractor_actor(observations=observations,
                                                              lstm_observations=lstm_observations,
                                                              lstm_states=lstm_states_actor)
            mid_features_critic, lstm_states_critic = self.feature_extractor_critic(observations=observations,
                                                              lstm_observations=lstm_observations,
                                                              lstm_states=lstm_states_critic)
            actions = self.actor(mid_features_actor)
            value = self.critic(mid_features_critic)

            # set variables not needed
            lstm_states = None
            if self.version == "mlp_separate":
                lstm_states_actor=None
                lstm_states_critic=None
        else:
            print("(BRAIN) Error, no valid version specified.")

        return actions, value, lstm_states, lstm_states_actor, lstm_states_critic

    def forward_pass(self, observations: torch.Tensor, lstm_observations: torch.Tensor = None,
                     actions=None, lstm_states=None, lstm_states_actor=None, lstm_states_critic=None, evaluation_mode=False):
        """

        @param observations:
        @param lstm_observations:
        @param actions:
        @param lstm_states:
        @param lstm_states_actor:
        @param lstm_states_critic:
        @param evaluation_mode:
        @return:
        """
        if self.version == "mlp_shared" or self.version == "mlplstm_shared": # shared feature extactor between actor and critic
            ### FEATURES EXTRACTION
            mid_features, lstm_states = self.feature_extractor(observations=observations,
                                                              lstm_observations=lstm_observations,
                                                              lstm_states=lstm_states)
            ### CRITIC
            # get value estimate from critic (value network) using these mid_features
            value_estimate = self.critic(mid_features)
            ### ACTOR
            # get estimated action means from actor (policy network) using these mid features
            action_means = self.actor(mid_features)

            # set variables not needed
            lstm_states_actor = None
            lstm_states_critic = None
            if self.version == "mlp_shared":
                lstm_states=None

        elif self.version == "mlp_separate" or self.version == "mlplstm_separate": # separate layers between actor and critic
            ### FEATURES EXTRACTION
            mid_features_actor, lstm_states_actor = self.feature_extractor_actor(observations=observations,
                                                                                lstm_observations=lstm_observations,
                                                                                lstm_states=lstm_states_actor)
            mid_features_critic, lstm_states_critic = self.feature_extractor_critic(observations=observations,
                                                                                   lstm_observations=lstm_observations,
                                                                                   lstm_states=lstm_states_critic)
            ### CRITIC
            # get value estimate from critic (value network) using these mid_features
            value_estimate = self.critic(mid_features_critic)
            ### ACTOR
            # get estimated action means from actor (policy network) using these mid features
            action_means = self.actor(mid_features_actor)

            # set variables not needed
            lstm_states = None
            if self.version == "mlp_separate":
                lstm_states_actor=None
                lstm_states_critic=None
        else:
            print("(BRAIN) Error, no valid version specified.")
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
            # because we don't want to sum action log probabilities over all days, but just the action log probabilities overall actions of ONE day
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

        else: # by default, forward pass
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
            return value_estimate, action_samples, actions_joint_log_proba, action_means, actions_distribution, stdev, \
                   lstm_states, lstm_states_actor, lstm_states_critic

    def predict(self, new_obs, new_lstm_obs=None, deterministic=False, lstm_states=None, lstm_states_actor=None, lstm_states_critic=None):
        """

        @param new_obs: New observation coming from environment used for predictions
        @param new_lstm_obs: New observation coming from environment but only for lstm
        @param deterministic: True by default. If True, will sample action from distribution defined by parameters estimated from the policy.
                              If False, will pick mean action.
        @param lstm_states: hidden and cell state tuple from forward pass from the previous state. encodes previous part of time series
                            only applicable if lstm net version SHARED chosen (so same lstm state for actor and critic, since shared feature extractor)
        @param lstm_states_actor: hidden and cell state tuple from forward pass from the previous state for the ACTOR. encodes previous part of time series
                            only applicable if lstm net version SEPARATE chosen
        @param lstm_states_critic: hidden and cell state tuple from forward pass from the previous state for the CRITIC. encodes previous part of time series
                            only applicable if lstm net version SEPARATE chosen
        @return:
        """
        # deterministic / non-deterministic:
        # https://datascience.stackexchange.com/questions/56308/why-do-trained-rl-agents-still-display-stochastic-exploratory-behavior-on-test
        # note: it somehow yields better result in testing when sampling (non-deterministic) actions than when I use the mean directly
        # this is probably because the probability distribution is part of the agent model and therefore model-leaning
        # predictions should be sampled

        # change new observation to torch tensor, if it is not already a torch tensor
       # if isinstance(new_obs, torch.Tensor):
        #    pass
        #else:
        #    new_obs = torch.as_tensor(new_obs, dtype=torch.float)
        # change new lstm observation to torch tensor of shape 3D, if it is not already a torch tensor
        #if isinstance(new_lstm_obs, torch.Tensor):
        #    if new_lstm_obs.dim() != 3: # we will only p
        #        new_lstm_obs.reshape((1, 1, 3)) # reshape to 3D, with (batch number,
        #else:
        #    new_obs = torch.as_tensor(new_obs, dtype=torch.float)

        ### FEATURES EXTRACTION
        # get intermediate features from features extractor
        if self.version == "mlp_shared" or self.version == "mlplstm_shared":  # shared feature extactor between actor and critic
            mid_features, lstm_states = self.feature_extractor(observations=new_obs,
                                                               lstm_observations=new_lstm_obs,
                                                               lstm_states=lstm_states)
            ### CRITIC
            # get value estimate from critic (value network) using these mid_features
            value_estimate = self.critic(mid_features)
            ### ACTOR
            # get estimated action means from actor (policy network) using these mid features
            action_means = self.actor(mid_features)

            #set variables not needed
            lstm_states_actor = None
            lstm_states_critic = None
            if self.version == "mlp_shared":
                lstm_states=None

        elif self.version == "mlp_separate" or self.version == "mlplstm_separate":  # separate layers between actor and critic
            mid_features_actor, lstm_states_actor = self.feature_extractor_actor(observations=new_obs,
                                                                                lstm_observations=new_lstm_obs,
                                                                                lstm_states=lstm_states_actor)
            mid_features_critic, lstm_states_critic = self.feature_extractor_critic(observations=new_obs,
                                                                                lstm_observations=new_lstm_obs,
                                                                                lstm_states=lstm_states_critic)
            ### CRITIC
            # get value estimate from critic (value network) using these mid_features
            value_estimate = self.critic(mid_features_critic)
            ### ACTOR
            # get estimated action means from actor (policy network) using these mid features
            action_means = self.actor(mid_features_actor)
            #set variables not needed
            lstm_states=None
            if self.version == "mlp_separate":
                lstm_states_actor=None
                lstm_states_critic=None
        else:
            print("(BRAIN) Error, no valid version specified.")

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
        predicted_value = value_estimate
        # note: prediction here is deterministic; we take the action means (= the output of the actor network)
        # we could also sample an action from our actor network instead (non-deterministic prediction),
        # but if we predict we normally want to use the most likely prediction of our current model for performance evaluation
        return predicted_actions, predicted_value, lstm_states, lstm_states_actor, lstm_states_critic