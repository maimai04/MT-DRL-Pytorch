import numpy as np
import math
import torch

# import own libraries
#try:
#    from config.config import *
#except:
#    from config import *

########################################################################
# CUSTOM ON POLICY ALGORITHM                                           #
########################################################################

class OnPolicyBuffer():
    """
    This class does the following:
      - store trajectories and the corresponding metrics collected by the agent:
        observations, actions (from forward pass on actor/policy network),
        log probabilities of actions, rewards, dones,
        value estimates (from forward pass on critic/value estimator network),
        calculated rewards-to-go,
        calculated advantage estimates
      - converts stored lists /arrays to tensorflow tensors
      - samples batches of data and returns them to the agent
    """

    def __init__(self,
                 buffer_size,
                 obs_shape, # must be tuple (observations,)
                 actions_number: int,
                 lstm_obs_shape=None,
                 lstm_hidden_size: int=32
                 ):
        self.buffer_size = buffer_size  # buffer size ALWAYS = length of the data set here
        self.obs_shape = obs_shape  # Tuple[int, int]
        self.lstm_hidden_size = lstm_hidden_size
        if lstm_obs_shape == None:
            self.lstm_obs_shape = obs_shape
        else:
            self.lstm_obs_shape = lstm_obs_shape

        self.actions_number = actions_number

        # observations come in as numerical list / array per step
        self.obs = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        # lstm observations (only observations which are input to the lstm, such as log returns and vix)
        self.lstm_obs = np.zeros((self.buffer_size,) + self.lstm_obs_shape, dtype=np.float32)
        # hidden states / lstm states to be saved
        # about the shapes: the array needs to be of length buffer_size, then it needs to be able to store
        # arrays of shape (2,1,32) = 2 arrays of shape 1*32 (with 32 being the lstm hidden size)
        self.lstm_state_h = np.zeros((self.buffer_size,) + (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_c = np.zeros((self.buffer_size,) + (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_actor_h = np.zeros((self.buffer_size,)+ (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_actor_c = np.zeros((self.buffer_size,)+ (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_critic_h = np.zeros((self.buffer_size,)+ (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_critic_c = np.zeros((self.buffer_size,)+ (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        # actions come in as tensor (estimation of actor) per step
        self.actions = np.zeros((self.buffer_size, self.actions_number), dtype=np.float32)
        self.actions_clipped = np.zeros((self.buffer_size, self.actions_number), dtype=np.float32)
        # actions log probabilities come in as list / array per step
        self.actions_log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        # value estimates come in as tensor (estimation of critic) per step
        self.value_estimates = np.zeros((self.buffer_size,), dtype=np.float32)
        # rewards come in as single numerical entry per step
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        # dones come in as single False / True entry per step
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)

        # advantage estimates come in as list / array, calculated at the end of the whole rollout
        self.advantage_estimates = np.zeros((self.buffer_size,), dtype=np.float32)
        # returns calculated using advantage estimates and value estimates (advantage + value = return)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        # create dict for storing trajectory elements based on keys
        self.trajectory_dict = {}
        self.trajectory_dict.update({"obs": self.obs,
                                     "actions": self.actions,
                                     "actions_clipped": self.actions_clipped,
                                     "actions_log_probs": self.actions_log_probs,
                                     "rewards": self.rewards,
                                     "dones": self.dones,
                                     "value_estimates": self.value_estimates,
                                     "advantage_estimates": self.advantage_estimates,
                                     "returns": self.returns,
                                     "lstm_obs": self.lstm_obs,
                                     "lstm_state_h": self.lstm_state_h,
                                     "lstm_state_c": self.lstm_state_c,
                                     "lstm_state_actor_h": self.lstm_state_actor_h,
                                     "lstm_state_actor_c": self.lstm_state_actor_c,
                                     "lstm_state_critic_h": self.lstm_state_critic_h,
                                     "lstm_state_critic_c": self.lstm_state_critic_c})

    def add(self, object_to_add, key_name, position=0):
        """
        Add trajectory elements to OnPolicyBuffer
        """
        ### note on tensors conversions: (from google)
        # clone() creates a copy of tensor that imitates the original tensor 's requires_grad field.
        # It enables us to the tensor while still keeping the copy as a part of the computation graph it came from
        # (e.g. this is what we want to do with everything that matters for the loss calculation, such as values, log.probabilities)
        # detach() should be used when we want to remove a tensor from a computation graph
        # (e.g. later for actions, because they are not relevant for loss calculation and backpropagation)
        if isinstance(object_to_add, torch.Tensor):
            # note: tf.identity() creates a copy of the tensor, it is the tf equivalent of e.g. np.copy() or pd.copy()
            if key_name == "value_estimates":
                self.trajectory_dict[key_name][position] = object_to_add.clone().numpy().flatten()  # flatten should only be applied to values, ont to log_prob
            elif key_name in ["lstm_state_h", "lstm_state_c", "lstm_state_actor_h", "lstm_state_actor_c",
                              "lstm_state_critic_h", "lstm_state_critic_c"]:
                self.trajectory_dict[key_name][position] = object_to_add
            else:
                self.trajectory_dict[key_name][position] = object_to_add.clone().numpy()
        else:
            self.trajectory_dict[key_name][position] = np.array(object_to_add).copy()

    def return_data(self, key_name=None):
        """
        Return whole saved trajectories batch.
        """
        if key_name == None:
            return self.trajectory_dict
        else:
            return self.trajectory_dict[key_name]

    def calculate_and_store_advantages(self, terminal_V_estimate, gamma, gae_lambda):
        """
        Calculate the GAE (Generalized Advantage Estimate)

        We need to calculate the advantage estimates because they are part of the surrogate loss
        function. If our gae lambda (smoothing factor) =1, then we simply get the "vanilla" version
        of the advantage; A(s)=R-V(s), with R= discounted bootstrapped reward
        (bootstrapped means that we use estimates to calculate the reward)
        see also: https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
        Note: there are also other ways to calculate the advantage estimate, however the paper uses this GAE
        (generalized advantage estimation) because it is an exponential smoothing of the vanilla version and hence it has lower variance,
        which is good.
        """
        # Take the terminal state value V estimate and convert it to a numpy array (it is an output of the network,
        # hence a tensor. Then we need to flatten it to go from format [V1],[V2],...] to [V1, V2,...]
        v_terminal = terminal_V_estimate.clone().numpy().flatten()
        # the last / next advantage s by definition 0, as is the last value (since both are defined by future (discounted) rewards,
        # and there are none after the end of the episode)
        next_advantage_value = 0
        # we are starting in reversed order (see also explanation in the thesis)
        for t in reversed(range(self.buffer_size)):
            # if we are at the last step in the trajectory, then the delta reduces to
            # reward of (state, action) @t (received at state t+1 but stored in the array at place t) - state value V @t
            # Note: "delta" is the TD error (=Temporal Difference) error
            if t == self.buffer_size - 1:  # Note: range starts counting with 0 and hence the largest and final t will be buffer_size-1
                not_terminal = 1 #0  # 0 stands for FALSE, so if we are at t==buffersize-1, we ARE in the terminal state and hence not:terminal = False
                next_state_value = v_terminal  # the next state value is hence the terminal state value
                # NOTE: at first, I had it "wrong" in the sense that I used not_terminal = 0, because every source I found online about explaining
                # how to code the GAE and every book always only covered EPISODIC tasks - tasks where a goal is reached /
                # the game is terminated because the agent dies etc. and the EPISODE ENDS => so it makes sense that the TRUE VALUE in the last state of an
                # episodic task is, by definition, 0 (since state value = present value of all future rewards).
                # However, this task in my work is a bit different: even though I call the data trajectories "episodes" (=abuse of language
                # because I named it like that in the beginning when I didn't know better), my task is a continuing RL problem,
                # that means that the final value is not really 0 => we don't suddenly loose all the money we have invested and we also don't receive a last,
                # large reward at the last time step. So basically, the agent here (should) assume(s) that after the trajectory (not really episode) ends,
                # the data ends but not the task. So therefore I changed the "mask" above to 1 as well, so that we use the final value estimate in order
                # to calculate the GAE. I haven't rigorously copared performances of algorithm with / vs. without GAE, but I have calculated it on a
                # mini sample for myself and I can tell you that the GAE is very different depending on whether we use the episodic or non-episodic way of calculation.
                # If we use episodic (with masking in last state to 0) GAE estimates, we get a more negative view of the return (but the effect diminishes when the
                # trajectory is long and the because of the discounting AND smoothing with the gae_lambda, the final value estimate doesn't have so much weight actually)
            # if we are not at the last step of the trajectory anymore: (that means t < buffer_size-1, starting with t=buffer_size-2)
            # (Note: "anymore", because we always first start at t==buffersize-1, at the end of the trajectory we have)
            else:
                not_terminal = 1  # 1 stands for TRUE
                # we get the stored state value from the buffer.
                # now, t<=buffer_size-2, and the value at t+1 = buffersize-2+1 = buffer_size-1 = last value stored in buffer
                # (note: the terminal value is not stored in the buffer, because we don't need it for the updates
                # (nothing comes after it), only for the advantage calculation)
                next_state_value = self.trajectory_dict["value_estimates"][t + 1]
                # now that we found out whether the next state is the terminal state or not, and now that we
            # got the value of the next state, we can compute the advantage of the current time point
            # DELTA is simply the TD (temporal difference) at time t
            # If we are at the final step (meaning the next state is the terminal state),
            # then we only get self.trajectory_dict["rewards"][t] - self.trajectory_dict["value_estimates"][t] (since_not_terminal = 0 (False))
            # Note that we have stored the trajectories as follows; (obs@t, action@t, reward (coming from (state,action)@t, received @t+1) is all stored in the array at position t,
            # hence reward[t] is the reward we receive at time step t+1 but we get it because we take a specific step at time t,
            # while the value[t] is the value of the state in which we take the action to get reward[t]. Therefore, reward[t]-value[t] (if we are in the last step) is the
            # temporal difference, because value is the estimate of all future resturns and here reward[t] is the only future return and therefore the value estimate
            # would be perfect if it would equal the reward received (ex ante, since now the probability of getting into this state and getting this reward is 100%)
            # gamma is the discount factor, gae_lambda the smoothing factor; we specify both in the PPO class object
            delta = self.trajectory_dict["rewards"][t] + (gamma * next_state_value * not_terminal) - \
                    self.trajectory_dict["value_estimates"][t]
            # compute the advantage value of the time step t. If we are in the last time step, after which only the terminal state comes, this
            # term reduces to delta only. Else, we smooth the advantage value of the next time step t with the gae_lambda, then discount it to time step t
            next_advantage_value = delta + gamma * (gae_lambda * not_terminal * next_advantage_value)
            # we add the calculated advantage to the buffer storage
            self.trajectory_dict["advantage_estimates"][t] = next_advantage_value

    def calculate_and_store_returns(self):
        # returns are calculated using the advantage estimates + our value estimate in order to get less noisy estimates of the returns than
        # if we would simply use the rewards.
        self.trajectory_dict["returns"] = self.trajectory_dict["advantage_estimates"] + self.trajectory_dict["value_estimates"]

    def reset(self):
        """
        We need to reset the buffer every time before we start collecting and storing experiences anew,
        otherwise we will keep the old data and just add new data and then it will be a mess.
        So we reset the buffer in order to "clear" it.
        """
        # observations come in as numerical list / array per step
        self.obs = np.zeros((self.buffer_size,) + self.obs_shape, dtype=np.float32)
        # lstm observations (only observations which are input to the lstm, such as log returns and vix)
        self.lstm_obs = np.zeros((self.buffer_size,) + self.lstm_obs_shape, dtype=np.float32)
        # hidden states / lstm states to be saved
        # about the shapes: the array needs to be of length buffer_size, then it needs to be able to store
        # arrays of shape (2,1,32) = 2 arrays of shape 1*32 (with 32 being the lstm hidden size)
        self.lstm_state_h = np.zeros((self.buffer_size,) + (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_c = np.zeros((self.buffer_size,) + (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_actor_h = np.zeros((self.buffer_size,)+ (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_actor_c = np.zeros((self.buffer_size,)+ (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_critic_h = np.zeros((self.buffer_size,)+ (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        self.lstm_state_critic_c = np.zeros((self.buffer_size,)+ (2,1) + (self.lstm_hidden_size,), dtype=np.float32)
        # actions
        self.actions = np.zeros((self.buffer_size, self.actions_number), dtype=np.float32)
        self.actions_clipped = np.zeros((self.buffer_size, self.actions_number), dtype=np.float32)
        # actions log probabilities come in as list / array per step
        self.actions_log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        # value estimates come in as tensor (estimation of critic) per step
        self.value_estimates = np.zeros((self.buffer_size,), dtype=np.float32)
        # rewards come in as single numerical entry per step
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        # dones come in as single False / True entry per step
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)

        # advantage estimates come in as list / array, calculated at the end of the whole rollout
        self.advantage_estimates = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        # create dict for storing trajectory elements based on keys
        self.trajectory_dict = {}
        self.trajectory_dict.update({"obs": self.obs,
                                     "actions": self.actions,
                                     "actions_clipped": self.actions_clipped,
                                     "actions_log_probs": self.actions_log_probs,
                                     "rewards": self.rewards,
                                     "dones": self.dones,
                                     "value_estimates": self.value_estimates,
                                     "advantage_estimates": self.advantage_estimates,
                                     "returns": self.returns,
                                     "lstm_obs": self.lstm_obs,
                                     "lstm_state_h": self.lstm_state_h,
                                     "lstm_state_c": self.lstm_state_c,
                                     "lstm_state_actor_h": self.lstm_state_actor_h,
                                     "lstm_state_actor_c": self.lstm_state_actor_c,
                                     "lstm_state_critic_h": self.lstm_state_critic_h,
                                     "lstm_state_critic_c": self.lstm_state_critic_c
                                     })


