import logging
import math
import os
import numpy as np
import pandas as pd
from itertools import chain
import gym
import torch
from gym import spaces
from torch import nn



class FinancialMarketEnv(gym.Env):
    """
    This class implements the stock trading environment for thiw work. It suclasses the gym.Env class from
    OpenAI Gym. Note that this just means that the class meets some basic Gym standards, such that it has
    certain methods mandatory implemeted, namely: step, reset, render

    Stable baselines has a guide of how to create a stock trading environment:
    A stock trading environment for OpenAI gym
    see also: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
    complete guide on how to create a custom env. with OpenAI gym :
              https://github.com/openai/gym/blob/master/docs/creating-environments.md

    This environment was initially inherited from:
    Yang et al. (2020): Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy

    Everything was changed except from the way the
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 # whole dataset to use in env: train, validation or test dataset
                 df: pd.DataFrame,
                 # version for how the actions should be converted to buy / sell actions (see config file)
                 step_version,
                 # number of stocks
                 assets_dim: int,
                 # how long the observation vector should be for one day;
                 # n_stocks*n_features_per_stock+ n_stocks(for saving asset holdings) + n_other_features (if any) + 1 (for cash position)
                 shape_observation_space: int,
                 # note: this is always passed even if we don't actually use the lstm
                 shape_lstm_observation_space: int,
                 # where results should be saved
                 results_dir: str,
                 # list of features names, single feature names and lstm feature names (optional, only if applicable) to be used
                 features_list: list = [],
                 single_features_list: list = [],
                 # note: this is always passed even if we don't actually use the lstm
                 lstm_features_list: list = [],
                 # day (index) from which we start
                 day: int = 0,
                 # whether it is the validation, train or test env; there are some minor differences,
                 # such as that to the validation env, we always start testing with 0 asset positions,
                 # while for e.g. testing, provide the previous state (if there already was a testing period), so that
                 # the agent continues testing from where he stopped last time before the roll-forward of the train / validation/ test windows
                 mode: str = "",  # "validation", "test" =trade
                 # name of the model, e.g. "ppo" (only relevant for saving)
                 model_name: str = "",
                 # iteration = the episode we are currently in, used for saving
                 iteration: int = 1,
                 # this is mutiplied by the actions given by the policy, which are usually between -10, 10,
                 # in order to get the number of assets to buy
                 hmax_normalize: int = 100,
                 # this is the starting cash balance. In this work, we are not allowed to short or to buy more than we can afford
                 initial_cash_balance: float = 1000000,
                 # this is a flat rate transaction cost which is multiplied by the trading volume
                 transaction_fee_percent: float = 0.001,
                 # if rewards are in absolute portfolio change values, they need to be scaled down a bit because they
                 # can be quite large, like 2000, this is better for convergence
                 reward_scaling: float = 1e-4,
                 # whether we are in the initial episode or not.
                 # if we are not in the initial episode, then for the test set we pass the previous state to it.
                 # for the train set, it depends whether we want to retrain or continue training with the saved model
                 initial: bool = True,
                 # previous state, if available
                 previous_state: list = [],
                 # previous asset price list, if available
                 previous_asset_price: list = [],
                 # name of the price column, here "adjcp"
                 price_colname: str = "adjcp",
                 # a counter for how often the env was reset, for analysis / debugging
                 reset_counter: int = 0,
                 # a counter of how often the agent reached the final state (= end of the provied train / validation / test dataset)
                 # also used for analysis / debugging
                 final_state_counter: int = 0,
                 # counter of how many steps were taken in one episode, used for saving results and for analysis / debugging
                 steps_counter: int = 0,
                 # whether we want to save results or not (default True, but for debugging sometimes False)
                 save_results=True,
                 calculate_sharpe_ratio=False,
                 logger=None,
                 # for special reward calculation based on sharpe ratio or semivariance penalty;
                 performance_calculation_window=7,
                 # penalty for rebalancing
                 rebalance_penalty: float=None,
                 # the measure for reward
                 reward_measure: str = "addPFVal",
                 total_episodes_to_train: int=10,
                 ):
        # we call the init function in the class gym.Env
        super().__init__()
        """
        @param df: pd.DataFrame(), sorted by date, then ticker, index column is the datadate factorized
                   (split_by_date function)
        ...
        """
        self.logger = logger
        self.reset_counter = reset_counter
        self.final_state_counter = final_state_counter
        self.steps_counter = steps_counter

        self.mode = mode
        self.df = df.copy()
        self.features_list = features_list
        self.single_features_list = single_features_list
        self.lstm_features_list = lstm_features_list
        self.firstday = day
        self.day = day
        self.model_name = model_name
        self.iteration = iteration

        self.initial = initial
        self.previous_state = previous_state
        self.previous_asset_price = previous_asset_price

        self.hmax_normalize = hmax_normalize
        self.initial_cash_balance = initial_cash_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling
        self.assets_dim = assets_dim
        self.shape_observation_space = shape_observation_space
        self.shape_lstm_observation_space = shape_lstm_observation_space

        self.price_colname = price_colname
        self.results_dir = results_dir
        self.save_results = save_results
        self.calculate_sharpe_ratio = calculate_sharpe_ratio
        self.performance_calculation_window = performance_calculation_window
        self.step_version = step_version
        self.rebalance_penalty = rebalance_penalty
        self.reward_measure = reward_measure
        # used for saving only at the end of training (else not enough storage on laptop)
        self.total_episodes_to_train = total_episodes_to_train

        ##### CREATING ADDITIONAL VARIABLES
        # action_space normalization and shape is assets_dim
        self.data = self.df.loc[self.day, :]  # includes all tickers, hence >1 line for >1 assets
        self.datadate = list(self.data["datadate"])[0]  # take first element of list of identical dates

        # we change the action space; instead from -1 to 1, it will go from 0 to 1
        # Note: this only affects the clipping we might do in the PPO agent.
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.assets_dim,))
        # Note: in the ensemble paper, they had an observation space from 0 to inf,
        # but this doesn't make sense since some values of their observation space were definitively <0.
        # So I don't know if stable baselines did some clipping on the observation space there which might have affected the performance.
        # I don't do any clipping on the observation space, hence these boundaries don't affect my algorithm;
        # still, to be exact, I have changed the space to [-inf: +inf]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.shape_observation_space,))

        ##### INITIALIZING VARIABLES
        # set terminal state to False at initialization
        # Note: this is the same variable as "done" or "dones", the so-called "mask" in reinforcement learning, just renamed
        self.terminal_state = False

        # initializing state for current day:
        # current cash balance is the initial cash balance
        self.current_cash_balance = self.initial_cash_balance
        # number of asset holdings and asset equity weights are each a vector of zeroes with the same length as number of assets
        # (one place for each asset)
        self.current_n_asset_holdings = [0] * self.assets_dim
        # weights are a bit "special"; because they change twice in a state transition;
        # first, when we rebalance our portfolio, we change our weights but with the old asset prices
        # second, when the we get the new day / state and observe the new asset prices, the (money-) weights of our assets change again
        # here, we only record the (money-)weights of each asset at the beginning of the day, meaning: after both changes mentioned above
        # so we start with weights of 0; the next state will be n_asset_holdings*new_asset_price / (equity portfolio value)
        self.current_asset_equity_weights_startofday = [0] * self.assets_dim
        # then it is also interesting to track how the weights of all stocks change compared to the whole pf value (including cash)
        # and how much weight cash has in the portfolio, hence we create a vector of zeroes of length n_assets + 1 (for cash)
        # the last list entry will be for cash
        self.current_cash_weight = 1
        self.current_all_weights_startofday = [0] * self.assets_dim + [self.current_cash_weight]

        # in order to have it simpler to query, I created a dictionary for the state, where all the things to be saved
        # are put in there and accessible by "keyname"
        # Note: in step_version of the paper, they use Cash position and number of asset holdings in the state vector.
        # in my own version (newNoShort), I use the cash weights and the asset weights instead, since my outputted actions are also
        # interpreted as weights and hence I think this makes more sense (see documentation in written part of thesis)
        if self.step_version == "paper":
            self.state = {"cash": [self.current_cash_weight],
                          "n_asset_holdings": self.current_n_asset_holdings}
        elif self.step_version == "newNoShort":
            self.state = {"cash_w": [self.current_cash_weight],
                          "asset_w": self.current_all_weights_startofday[:-1]} # adding all but cash weight (which is the final entry)

        # if we are at the first time step of the episode, we get the asset names (don't need to do this at every step, but we could)
        if self.steps_counter == 0:
            self.asset_names = df["tic"].unique()
        # if we are at the first iteration (= first episode), and in the first step,
        # (we start counting episodes from 1, steps from 0, just because it was more practical for printing the number of episodes)
        # we create the state header for later being able to save all the states in one dataframe together with the header
        if self.iteration == 1 and self.steps_counter == 0:
            if self.step_version == "paper":
                self.state_header = ["cash"] + [s + "_n_holdings" for s in self.asset_names]
            elif self.step_version == "newNoShort":
                self.state_header = ["cash_w"] + [s + "_w" for s in self.asset_names]

        # for each feature in the features list we passed in the init, we update the state dictionary for this current day;
        # we update the state dict with the features
        for feature in self.features_list:
            self.state.update({feature: self.data[feature].values.tolist()})
            # if we are in the first iteration (=episode) and at the first step,
            # we add feature names with a suffix to th state header from before
            if self.iteration == 1 and self.steps_counter == 0:
                suffix = "_" + feature
                self.state_header += [s + suffix for s in self.asset_names]

        for feature in self.single_features_list:
            # for features which are not attached to a stock, like the vix (vixDiv100),
            # we don't want them to appear n_asset times in the state
            self.state.update({feature: [self.data[feature].values[0]]})
            if self.iteration == 1 and self.steps_counter == 0:
                self.state_header += ["vixDiv100"]

        # create lstm state as a subset of the state
        self.lstm_state = dict((k, self.state[k]) for k in self.lstm_features_list if k in self.state)
        self.lstm_state_flattened = np.asarray(list(chain(*list(self.lstm_state.values()))))
        # create flattened state_flattened (because we need to pass the sttae as np.array to the model
        # so it can create a torch tensor (cannpt use a dict)
        self.state_flattened = np.asarray(list(chain(*list(self.state.values()))))

        # initialize reward; update after each step (after new day sampled)
        self.reward = 0
        # initialize transaction cost; update after each trade
        self.cost = 0
        # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
        self.buy_trades = 0
        self.sell_trades = 0

        ##### INITIALIZE MEMORIES / MEMORY TRACKERS
        # note: key names must be the same as for paths.SUBBDIR_NAMES,
        # else saving to csv doesn't work (dirs won't match)
        self.memories = {"datadates": [self.datadate],
                         "cash_value": [self.initial_cash_balance],
                         "portfolio_value": [self.initial_cash_balance],
                         # starting value (beginning of the day), before taking a step
                         "rewards": [self.reward],
                         # reward in first entry is 0 because we can only calculate it after a day has passed (but could be done differnetly, doesn't matter)
                         "policy_actions": [],
                         "policy_actions_trans": [],
                         "exercised_actions": [],
                         "transaction_cost": [],
                         "number_asset_holdings": [self.current_n_asset_holdings],
                         "asset_equity_weights": [self.current_asset_equity_weights_startofday], # weight of each asset on the equity-only portion
                         "all_weights_cashAtEnd": [self.current_all_weights_startofday],
                         "sell_trades": [],
                         "buy_trades": [],
                         "state_memory": [self.state_flattened],
                         }
        # if we are in the first iteration (=episode), and the first step of the episode,
        # we pass the state header as vector to the memory dictionary,
        # and we will use it later for saving results
        if self.iteration == 1 and self.steps_counter == 0:
            self.memories.update({"state_header": self.state_header})

    def step(self, actions) -> list:
        """
        This function is used for the agent to take a step in the environment and hence to make a
        transition from one state to the other state, using actions provided by the RL agent.

        First, we check if we are in the terminal state of the episode (done=True) or not.
            If we are in the terminal state, we don't take any actions anymore, but just return the final state and reward,
                    and save the results (memory and state vector) to csv files
            If we are not in the terminal state, we actually take a step using the actions provided by the agent,
                    then the environment samples and returns the next state and a reward

        methods:
        _save_results()     : saving results to csv at the end of the episode, if terminap=True
        _terminate()        : this function is called if we are in the terminal state of the episode;
                              we then append the last data to the memory dictionary and call the _save_results() function
                              (if save_results == True).
                              Output:
                              self.state_flattened: np.array, self.reward: float, self.terminal_state: bool, {}
        _calculate_reward   : Inputs: end_portfolio_value: float=None, begin_portfolio_value: float=None
                              This function calculates the reward, depending on the calculation method specified
                              in config.py under settings.REWARD_MEASURE.

        @param actions: np.array of actions, provided by the agent
        @return: self.state: np.array, self.reward: float, self.terminal_state: bool, {}
        """
        def _get_sharpe_ratio(window="full") -> float: # todo: rm
            """
            This function is used to calculate the sharpe ratio for the current episode either at the
            end of the episode (for hyperparameter tuning) or at the end of each day if
            reward measure = Sharpe ratio.
            """
            dates = pd.DataFrame({"datadate": self.memories["datadates"]})
            pfval = pd.DataFrame(self.memories["portfolio_value"])
            df = pd.concat([dates, pfval], axis=1)
            df["datadate"] = pd.to_datetime(df["datadate"], format='%Y%m%d')
            df.columns = ["datadate", "pfvalue"]
            if window == "full":
                pass
            else:
                print(f"sharpe ratio, window {window}")
                df = df[-window:]
            # then we can create a "perf" object (performances) with the function .calc_stats() (bt = backtest)
            perf = df.set_index("datadate")["pfvalue"].calc_stats()
            sharpe_ratio_daily_ann = perf.daily_sharpe
            return sharpe_ratio_daily_ann

        def _get_semivariance(window="full") -> float:
            # calculated base on this formula: https://www.investopedia.com/terms/s/semivariance.asp
            dates = pd.DataFrame({"datadate": self.memories["datadates"]})
            pfval = pd.DataFrame(self.memories["portfolio_value"])
            df = pd.concat([dates, pfval], axis=1)
            df["datadate"] = pd.to_datetime(df["datadate"], format='%Y%m%d')
            df.columns = ["datadate", "pfvalue"]
            if window == "full":
                pass
            else:
                # get df according to window
                if self.steps_counter < self.performance_calculation_window-1:
                    print(f"semivariance, window {window}")
                df = df[-window:]
            # calculate the daily return for each day in the episode, while the first will naturally be Nan and we remove it
            daily_ret = df["pfvalue"].pct_change(1)[1:]
            # calculate the mean of all returns
            mean = np.mean(daily_ret)
            # get the daily returns which are below the mean of returns
            daily_ret_below_mean = daily_ret[daily_ret < mean]
            # subtract returns below the mean from the mean, and square each of these differences, then get the average
            semivariance = np.mean((mean - daily_ret_below_mean)**2)
            return semivariance

        def _save_results() -> None:
            # SAVING MEMORIES TO CSV
            # save dates vector alone to results (for debugging, analysis, plotting...)
            dates = pd.DataFrame({"datadate": self.memories["datadates"]})
            #dates.to_csv(os.path.join(self.results_dir, "datadates",
            #                            f"datadates"
            #                            f"_{self.mode}"
            #                            f"_{self.model_name}"
            #                            f"_ep{self.iteration}" # episode
            #                            f"_totalSteps_{self.steps_counter}"
            #                            f"_finalStateCounter_{self.final_state_counter}"
            #                            f".csv"))
            # if we are at the last step of the first episode, save header (stays the same for whole run,
            # # so no need to save again every time)
            if self.iteration == 1:
                pd.DataFrame(self.memories["state_header"]).\
                        to_csv(os.path.join(self.results_dir, "state_memory", "state_header.csv"))

            # in test mode, we only test once on the data, so we save always.
            # if we are not in test mode (train, validation), we train / validate for self.total_episodes_to_train times,
            # which can be a lot to save and we don't actually want all those intermediate .csv files (I don't even have enough free storage on my laptop)
            # so we only save at the end of training / validation and in the beginning (for debugging purposes)
            if self.mode == "test" or self.final_state_counter > self.total_episodes_to_train:
                # then, for each data entry in the memories dictionary, we save together with the corresponding dates to csv
                for key in list(self.memories.keys()):
                    # create pandas df for each key (except for datadate and state_header, because we already saved them before
                    if key not in ["datadates", "state_header"]:
                        keydf = pd.DataFrame(self.memories[key])
                    # "try" allows us to "try if the following operation works", and if it doesn't, there is no error,
                    # instead the "exception" is called (except) and the program can continue to run.
                    try:
                        # if each element in a key is a list and of same length as the asset names list,
                        # we know that each list element belongs to one of the assets and want to name each column after one.
                        # we use the function "try", because the comparison does not work if it is not a list / array.
                        if len(self.memories[key][0]) == len(self.asset_names):
                            keydf.columns = self.asset_names
                        # if each element i a key has the same length as asset names + 1 then the key is going to be the
                        # asset + cash weights key, so we want the header to be asset names + cash
                        if len(self.memories[key][0]) == len(self.asset_names)+1:
                            keydf.columns = list(self.asset_names) + ["Cash"]
                    except:
                        pass
                    # if the key is "state_memory" and we are in the first episode,
                    # we want the state_header to be the column names of the state memory
                    if key == "state_memory" and self.iteration == 1:
                        keydf.columns = self.memories["state_header"]
                    # concatenate each key df (except for "datadate" and "state header", which were saved separately)
                    # with the dates df and save to csv
                    if key not in ["datadates", "state_header"]:
                        pd.concat([dates, keydf], axis=1).to_csv(os.path.join(self.results_dir, key,
                                                                          f"{key}"
                                                                          f"_{self.mode}"
                                                                          f"_{self.model_name}"
                                                                          f"_ep{self.iteration}"
                                                                          f"_totalSteps_{self.steps_counter}"
                                                                          f"_finalStateCounter_{self.final_state_counter}.csv"))
            # save rewards separately because we want to save all validation rewards so we can later plot rewards vs. timesteps
            if self.mode == "validation":
                keydf = pd.DataFrame(self.memories["rewards"])
                pd.concat([dates, keydf], axis=1).to_csv(os.path.join(self.results_dir, "rewards",
                                                                      f"rewards"
                                                                      f"_{self.mode}"
                                                                      f"_{self.model_name}"
                                                                      f"_ep{self.iteration}"
                                                                      f"_totalSteps_{self.steps_counter}"
                                                                      f"_finalStateCounter_{self.final_state_counter}.csv"))

            return None

        def _terminate() -> list:
            self.final_state_counter += 1
            self.memories["exercised_actions"].append([0] * self.assets_dim)  # because no actions exercised in this last date anymore
            self.memories["transaction_cost"].append(0)  # since no transaction on this day, no transaction cost
            self.memories["sell_trades"].append(0)
            self.memories["buy_trades"].append(0)
            # SAVING MEMORIES TO CSV
            if self.save_results == True:
                _save_results()
            sharpe_ratio = {}
            if self.calculate_sharpe_ratio == True: # todo: rm
                sharpe_ratio = _get_sharpe_ratio()

            return [self.state_flattened, self.lstm_state_flattened, self.reward, self.terminal_state, sharpe_ratio]

        def _calculate_reward(end_portfolio_value: float=None, begin_portfolio_value: float=None):
            if self.reward_measure == "addPFVal":
                self.reward = end_portfolio_value - begin_portfolio_value
                # apply reward scaling to make rewards smaller
                self.reward = self.reward * self.reward_scaling

            elif self.reward_measure == "SR7": # 7 day sharpe ratio (non-annualized) # todo: rm
                if self.steps_counter >= self.performance_calculation_window-1:
                    self.reward = _get_sharpe_ratio(window=self.performance_calculation_window)
                else:
                    self.reward = np.log(end_portfolio_value / (begin_portfolio_value + 1e-00001)) # added very small number to prevent div. by 0

            elif self.reward_measure == "logU":
                # log Utility(new F value / old PF value) as proposed by Neuneier 1997
                self.reward = np.log(end_portfolio_value / (begin_portfolio_value + 1e-00001))

            elif self.reward_measure == "semvarPenalty":
                self.reward = np.log(end_portfolio_value / (begin_portfolio_value + 1e-00001))
                if self.steps_counter >= self.performance_calculation_window-1:
                    #print(f"steps: {self.steps_counter}")
                    semivar = _get_semivariance(window=self.performance_calculation_window)
                    log_U = np.log(end_portfolio_value / (begin_portfolio_value + 1e-00001))
                    self.reward = log_U - semivar
                    #print(f"semivariance: {semivar}")
                    #print(f"logU: {log_U}")
                else:
                    self.reward = np.log(end_portfolio_value / (begin_portfolio_value + 1e-00001))
            else:
                print("ERROR, no valid reward specified.")
            #print("reward: ", self.reward)
            return None

        def _take_step(actions: np.array) -> list:
            # we take a step in the environment, so we need to update the steps counter by 1 step
            self.steps_counter += 1

            ### GET DATA FROM BEFORE TAKING A STEP, WE WILL NEED IT LATER TO CALCULATE STUFF
            # we need the beginning values of cash, number of assets holdings these will change if we buy / sell stocks
            # we also need current asset prices (will change when new day is sampled)
            # and the current portfolio value (as a function of cash + asset_prices* n_asset_holdings)
            if self.step_version == "paper":
                # in the paper version, we have stored the cash value directly in the state vector
                begin_cash_value = self.state["cash"][0]
                begin_n_asset_holdings = self.state["n_asset_holdings"]

            elif self.step_version == "newNoShort":
                # in the custom version, we take the values from the memory instead, because the cash value and asset
                # holdings are not in the state vector
                begin_cash_value = self.memories["cash_value"][-1]
                begin_n_asset_holdings = self.memories["number_asset_holdings"][-1]

            # asset pricing at the beginning of the day, before taking the action
            begin_asset_prices = self.data[self.price_colname].values.tolist()
            # portfolio value in the beginning, before taking the action
            begin_portfolio_value = begin_cash_value + sum(np.array(begin_asset_prices) * np.array(begin_n_asset_holdings))

            ################################################################################################
            ### TAKE A STEP,
            ### (WITHIN THE CURRENT STATE, BEFORE THE ENVIRONMENT SAMPLES A NEW STATE)
            ################################################################################################
            # and we get the "indices" of at which position in the original actions array the sorted actions were
            if self.step_version == "paper":
                # So the actions from the model come out in the range of, like, -2, ..., +2 plus/minus
                # and then they are reshaped to the action space we have defined in the environment (reshaping is done in the ppo algorithm)
                # so that there will be no out-of-bound errors
                # and then here in this setup, the actions are multiplied by hmax, so that we get "number of stocks to buy or sell"
                actions = actions * self.hmax_normalize
                # this is the version from the ensemble paper.
                # they take an actions vector where each element can be between -1 and 1, which represents the number of
                # stocks to buy (since numbers are between -1 and 1, they will be multiplied in the buy and sell functions
                # by HMAX_NORMALIZE (which is 100 by default), hence the number of stocks we can buy and sell each time is limited to max. 100 resp. -100,
                # but in this version there also is no short-selling allowed.
                # by this amount. When I run this method, it often leads to a problem:
                # after a while, the cash is mostly distributed among stocks, and some stocks get expensive over time,
                # while some others don't so much, and the agent has prblems rebalancing because not enough cash is freed up
                # due to the nature of how actions are exercised. Another side effect is that
                # in the beginning, not all money is distributed among stocks (because one can only buy at max. 100 stocks of each stock),
                # so the entry into the market is more "careful"; basically, all the time, the nature of this rebalancing method
                # acts lik a rebalancing constraint, but I am not sure if it acts in the way we would want.
                # If, for example, one compares policy actions and exercised actions of later episodes of the test set,
                # we see that lillte of the policy actions are actually exercised, which is bad; because the agent only gets feedback
                # via reward, and the reward depends on: whether the weights the agent chose could be exercised, market conditions we could
                # "foresee" in the data (=> things the agent could learn), market conditions we cannot foresee and that make our portfolio bad even
                # though the agent did nothing wrong
                # and while the last thing, we cannot reall ydo something about, it is always good to let the "information gap" between agent and what comes out
                # of the action be as small as possible, hence the agent does not "know" how to attribute a reward (good weights, market luck, god weights but could not exercise)

                # we sort the actions such that they are in this order [-large, ..., +large]
                # then they sort into sell and buy actions
                argsort_actions = np.argsort(actions)
                # get assets to be sold (if action -)
                sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
                # get assets to be bought (if action +)
                buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
                # create empty list which will be filled with sell / buy actions for each asset
                exercised_actions = [0] * self.assets_dim

                # Note: this iterates index by index (stock by stock) and sells / buys stocks in consecutive order.
                # for buy: can it happen that one stock uses up all ressources and the others then cannot be bought anymore, in the extreme case
                # also, one cannot buy /sell more than 100 stocks at each time step and the cash for buying is also limited
                for index in sell_index:  # index starting at 0
                    exercised_actions = self._sell_stock(index, actions[index], exercised_actions)
                for index in buy_index:
                    exercised_actions = self._buy_stock(index, actions[index], exercised_actions)

            elif self.step_version == "newNoShort":
                # in this version, actions represent target portfolio weights and need to be converted to actual "actions"
                # no short selling allowed
                # weights are received as a vector where values are in the interval [0,1] (softmax), and they sum up to one (L1 norm)
                # this is done like this: 1. target weight is compared with current weight.
                # delta_weight is calculated considering transaction cost (actual_w-target_w)*(1-transaction_cost)
                # assumption / constraint: every buy / sell action must be self-financing
                # that also means: if it is a sell action, we sell delta_weight, in order to free up money (we will then get sell_amount*(1-tc))
                # if it is a buy action, we buy delta_weight(1-transaction_cost)
                # in both cases we consider the constraint that we cannot sell / buy fractions of shares, and we cannot short-sell in this version,
                # hence we can only:
                # sell int(min(delta_weight*portfolio_value, number_assets_held_at_the_moment)) # note we round down, but could be done differently.
                # but since we are going easier on selling (sell action not dependent on transaction cost), we will likely end up being able to do
                # the whole sell action more often than to do the buy action, and then it is ok to round down the number of stocks to sell, if it is a float.
                # we can only:
                # buy int(delta_weight*portfolio_value*(1-transaction_cost)), so if we have int(5.6 stocks to buy) =>  stocks to buy,
                # here the idea is that we only want to invest delta_weight*portfolio_value money in this stock, and with transaction cost,
                # we end up getting only 5.6 stocks with that amount, so we take 5.
                # again, this could be done differently, depends on what we want. But like this, we are sure to never run out of cash,
                # and also, we have some spare cash freed uü in case we want to rebalance in the next period.
                target_weights = actions

                # rebalancing weight = new weights - current weights (weights taken as asset weights / all asset value, including cash,
                # # because otherwise we would end up never buying anything (because initially, all stocks value = 0, cash value = pf value = 1000000)
                # [-1]: we take the last list / array that was appended
                # [:-1], we take all the stock weights except for the cash weight (which is at the end, as the key name indicates)
                # delta_weight = new_weight (vector) - old_weights (vector)
                delta_weight = target_weights - self.memories["all_weights_cashAtEnd"][-1][:-1]

                #print("target_weights")
                #print(target_weights)
                #print("self.memories[all_weights_cashAtEnd][-1][:-1]")
                #print(self.memories["all_weights_cashAtEnd"][-1][:-1])

                # split weights into sell and buy actions (if -, sell; if +, buy)
                argsort_delta_weight = np.argsort(delta_weight)
                # get assets to be sold (if action -)
                sell_index = argsort_delta_weight[:np.where(delta_weight < 0)[0].shape[0]]
                # get assets to be bought (if action +)
                buy_index = argsort_delta_weight[::-1][:np.where(delta_weight > 0)[0].shape[0]]
                # create empty list which will be filled with sell / buy actions for each asset
                exercised_actions = [0] * self.assets_dim
                # max_cash available for buying; we will use this to adjust the buy-delta_weights
                # in the buy action because sometimes, if we cannot sell the whole sell-amount, we don't free up
                # enough cash and we might not have any spare cash so we cannot do the whole buy-action (invest the whole
                # max_cash_to_distribute_for_buying amount. Else we would get negative cash positions (most of the time)
                # and we don't simply want to buy stocks in the index order because then the last stocks might end up
                # never having spare money to be bought.

                #print("delta w")
                #print(delta_weight)
                #print("pf val")
                #print(self.memories["portfolio_value"][-1])

                max_cash_to_distribute_for_buying_after_tc = sum(delta_weight[delta_weight > 0]) \
                                                        * self.memories["portfolio_value"][-1] * \
                                                        (1-self.transaction_fee_percent)
                #print("delta_weight")
                #print(delta_weight)

                # Note:
                # max_cash_to_distribute_for_buying = delta_weight_to_buy * PF_value * (1-tc)
                # free_cash_to_distribute_for_buying (after selling action) = delta_weight_to_buy_NEW * PF_value * (1-tc)
                #   = total cash available at this time
                # to find delta_weight_to_buy_NEW:
                # sum(delta_weight_to_buy_NEW) = (free_cash_to_distribute_for_buying / PF_Value) * (1/(1-tc))
                # insert PF_Value
                # delta_weight_to_buy_NEW = (free_cash_to_distribute_for_buying /
                #                                 (max_cash_to_distribute_for_buying / (sum(delta_weight_to_buy) (1-tc)))
                #                                 * (1/(1-tc))
                # delta_weight_to_buy_NEW = (free_cash_to_distribute_for_buying / max_cash_to_distribute_for_buying)
                #                                 * (delta_weight_to_buy * (1-tc)) * (1/(1-tc))
                # delta_weight_to_buy_NEW = (free_cash_to_distribute_for_buying / max_cash_to_distribute_for_buying)
                #                                 * delta_weight_to_buy

                for index in sell_index:  # index starting at 0
                    exercised_actions = self._sell_stock(index, delta_weight[index], exercised_actions)

                free_cash_to_distribute_for_buying_after_tc = self.memories["cash_value"][-1] * (1 - self.transaction_fee_percent)
                #if self.mode == "train" and self.steps_counter <= 3:
                    #print("cash")
                    #print(self.memories["cash_value"][-1])
                    #print("\n---max_cash_to_distribute_for_buying")
                    #print(max_cash_to_distribute_for_buying_after_tc)
                    #("---free_cash_to_distribute_for_buying")
                    #print(free_cash_to_distribute_for_buying_after_tc)
                delta_weight_new = (min(free_cash_to_distribute_for_buying_after_tc, max_cash_to_distribute_for_buying_after_tc) / \
                                    max_cash_to_distribute_for_buying_after_tc) * delta_weight
                #print("delta_weight_new")
                #print(delta_weight_new)
                for index in buy_index:
                    exercised_actions = self._buy_stock(index, delta_weight_new[index], exercised_actions)

            ### UPDATE VALUES AFTER ACTION TAKEN (WITHIN THE CURRENT STATE, BEFORE THE ENVIRONMENT SAMPLES A NEW STATE)
            # after we took an action, what changes immediately and independently of the next state we will get from
            # the environment, are the number of asset holdings, the new cash balance and
            if self.step_version == "paper":
                self.current_cash_balance = self.state["cash"][0]
                self.current_n_asset_holdings = list(self.state["n_asset_holdings"])
            elif self.step_version == "newNoShort":
                self.current_cash_balance = self.memories["cash_value"][-1]
                self.current_n_asset_holdings = self.memories["number_asset_holdings"][-1] # note: current asset holdings is a list, so indexing is correct

            # append new data after taking the actions
            # this is the data that doesn't change based on the market condition that comes with the next day
            self.memories["exercised_actions"].append(exercised_actions)
            self.memories["sell_trades"].append(self.sell_trades)
            self.memories["buy_trades"].append(self.buy_trades)
            self.memories["transaction_cost"].append(self.cost)
            self.memories["number_asset_holdings"].append(self.current_n_asset_holdings)
            self.memories["cash_value"].append(self.current_cash_balance)

            ################################################
            ### ENVIRONMENT SAMPLES A NEW DAY, NEW STATE   #
            ################################################
            # update the day index by 1
            self.day += 1
            # get new day (date, stock prices, other features) from provided dataset (train, validation, test)
            self.data = self.df.loc[self.day, :]

            ### UPDATE VALUES FOR NEW DAY
            # get list of new dates and append to the memory dictionary
            self.datadate = list(self.data["datadate"])[0]
            self.memories["datadates"].append(self.datadate)

            # get asset prices of new day
            current_asset_prices = self.data[self.price_colname].values.tolist()
            # calculate current portfolio value, after taking a step (which changed the cash balance and number of asset holdings / pf weights)
            # and after getting a new day sample from the environment (with new asset prices)
            current_portfolio_value = self.current_cash_balance + sum(np.array(self.current_n_asset_holdings) *
                                                                  np.array(current_asset_prices))
            # weights of each asset in terms of equity pf value = number of assets held * asset prices / equity portfolio value
            self.current_asset_equity_weights_startofday = np.array(self.current_n_asset_holdings) * np.array(current_asset_prices) / \
                                                           (current_portfolio_value - self.current_cash_balance)
            # eights of each asset and of cash in terms of total portfolio value = number of assets held * asset prices / total pf value, then append cash weight at the end of the list
            self.current_cash_weight = self.current_cash_balance / current_portfolio_value
            self.current_all_weights_startofday = list(np.array(self.current_n_asset_holdings) * np.array(current_asset_prices) / \
                                                        current_portfolio_value) + [self.current_cash_weight]
            # get new asset prices
            self.observed_asset_prices_list = self.data[self.price_colname].values.tolist()
            # create new state dictionary; append current cash position and current asset holdings (after taking a step)
            if self.step_version == "paper":
                self.state = {"cash": [self.current_cash_balance],
                              "n_asset_holdings": self.current_n_asset_holdings}
            elif self.step_version == "newNoShort":
                self.state = {"cash_w": [self.current_cash_weight],
                              "asset_w": self.current_all_weights_startofday[:-1]}

            # for each provided feature, append the feature value for the new day to the state dictionary
            for feature in self.features_list:
                self.state.update({feature: self.data[feature].values.tolist()})

            for feature in self.single_features_list:
                # for features which are not attached to a stock, like the vix (vixDiv100),
                # we don't want them to appear n_asset times in the state
                self.state.update({feature: [self.data[feature].values[0]]})

            # create lstm state
            self.lstm_state = dict((k, self.state[k]) for k in self.lstm_features_list if k in self.state)
            self.lstm_state_flattened = np.asarray(list(chain(*list(self.lstm_state.values()))))
            # create flattened state (because we need to pass a np.array to the model which then converts it to a tensor,
            # cannot use a dictionary; but the dictionary is practical for querying and not loosing the overview of what is where)
            self.state_flattened = np.asarray(list(chain(*list(self.state.values()))))

            ### CALCULATE THE REWARD, FOLLOWING THE PREVIOUS STATE, ACTION PAIR
            _calculate_reward(end_portfolio_value=current_portfolio_value,
                              begin_portfolio_value=begin_portfolio_value)
            #self.reward = end_portfolio_value - begin_portfolio_value

            # Append rewards, new portfolio value and the new complete state vector to the memory dict
            self.memories["rewards"].append(self.reward)
            self.memories["portfolio_value"].append(current_portfolio_value)
            self.memories["state_memory"].append(self.state_flattened)
            self.memories["asset_equity_weights"].append(self.current_asset_equity_weights_startofday)
            self.memories["all_weights_cashAtEnd"].append(self.current_all_weights_startofday)

            ### RESET SOME COUNTERS
            # we want to get the transaction cost, sell trades and buy trades accumulated daily only (on a per step basis)
            # not accumulated over multiple days / steps
            self.cost = 0
            self.sell_trades = 0
            self.buy_trades = 0
            return [self.state_flattened, self.lstm_state_flattened, self.reward, self.terminal_state, {}]


        ####################################
        #    MAIN CODE FOR STEP FUNCTION   #
        ####################################
        # actions are in this form: [[1,2,3,4..]], need to reshape to [1,2,3,4,...]
        actions = actions.flatten()
        # save policy actions (actions) to memories dict (these are the actions given by the agent, not th actual actions taken)
        self.memories["policy_actions"].append(actions)
        # FIRST: CLIP / TRANSFORM ACTIONS to the range allowed by our objective /resp. the action space
        # this needs to be done because actions are sampled from a Distribution (here: Gaussian,
        # but could also use other, like e.g. Beta distribution), and that leads to actions not
        # necessarily being within the boundaries of the defined action space (note: I am using gym.Box for action space)
        # clipping is used by most implementations online, including stable baselines
        if self.step_version == "paper":
            # clip actions to be between action space limits (in paper [-1,1]
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        elif self.step_version == "newNoShort":
            # apply softmax again in sampled actions in order to make a vector of target weights
            # which are all between [0,1] and sum up to one together.
            actions = nn.functional.softmax(torch.as_tensor(actions, dtype=torch.float), dim=0).numpy()
        # save the clipped / transformed policy actions
        self.memories["policy_actions_trans"].append(actions)

        # if the sampled day index is larger or equal to the last index available in the data set provided,
        # we have reached the end of the episode.
        # Note: there are many ways to define an end of the episode, like when a certain reward / penalty is reached,
        # or we could also define fixed intervals for eth episodes, like 30 days etc., but here it makes sense to treat this
        # as if the episode is as long as the training data available, because theoretically it is a continuous problem,
        # where one invests over multiple years, theoretically "with no fixed end"
        self.terminal_state = self.day >= self.df.index.unique()[-1] # :bool

        ##### IF WE ARE IN THE TERMINAL STATE #####
        if self.terminal_state:
            # we call the function to terminate the episode, no step will be taken anymore
            return _terminate()
        ##### IF WE ARE NOT YET IN THE TERMINAL STATE #####
        else:
            # we take a step using the policy actions
            return _take_step(actions=actions)

    def _sell_stock(self, index, action, exercised_actions) -> list:

        def _sell_paper(index, action, exercised_actions) -> list:
            """
            If we hold assets, we can sell them based on our policy actions, but under short-selling constraints.
            Hence, we cannot sell more assets than we own.
            If we don't hold any assets, we pass, since we cannot short-sell.
            """
            # if we hold assets of the stock (given by the index), we can sell
            if self.state["n_asset_holdings"][index] > 0:
                # Note: I have changed everywhere into round() because else actions (n.stocks to buy) would be floats!
                # also: changed how state is constructed, using dict instead (not indizes)
                # based on short-selling constraints, get actually exercisable action based on policy action and current asset holdings
                # there is a difference between policy actions (actions given by the policy) and actually doable actions
                # we can sell the minimum between the number of stocks to sell given by the policy action,
                # and the number of assets we hold, since we don't allow short-selling
                exercised_action = min(round(abs(action)), self.state["n_asset_holdings"][index])
                exercised_actions[index] = -exercised_action
                # update cash balance; cash new = cash old + price * n_assets sold*(1 - transaction cost)
                self.state["cash"][0] += self.data[self.price_colname].values.tolist()[index] * \
                                         exercised_action * (1 - self.transaction_fee_percent)
                self.state["n_asset_holdings"][index] -= exercised_action
                # update transaction cost; cost + new cost(price * n_assets_sold * transaction fee)
                self.cost += self.data[self.price_colname].values.tolist()[index] * \
                             exercised_action * self.transaction_fee_percent
                # update sell trades counter
                self.sell_trades += 1
            # if we hold no stocks of the currently chosen stock (given by index), we cannot sell
            else:
                # we then do not exercise any actions (0 = hold position)
                exercised_actions[index] = 0
                pass
            # we return the list of exercised actions
            return exercised_actions

        def _sell_newNoShort(index, action, exercised_actions) -> list:
            # action is the delta_weight (-)
            delta_weight = action
            delta_weight_tc = delta_weight * (1-self.transaction_fee_percent) * (1 - self.rebalance_penalty)

            # now we convert this to the number of money to desinvest
            # note: the weight_X is defined as "value of all stock X holdings" / "total pf value (incl.cash)"
            # and delta_weight_X is the (sell) action we take in order to get to the "target value of all stock X holdings" / "total pf value (incl. cash)"
            money_to_desinvest = abs(delta_weight_tc) * self.memories["portfolio_value"][-1]


            # number of assets to sell is then simply "money to desivnest in stock X" // "price of a stock X"
            max_n_assets_to_sell = math.ceil(money_to_desinvest / self.data[self.price_colname].values.tolist()[index])
            n_assets_to_sell = min(max_n_assets_to_sell, self.memories["number_asset_holdings"][-1][index])
            #if self.mode == "train" and self.steps_counter <= 3 and index == 1:
                #print("(sell) price to get if delta-weight used: ")
                #print((abs(delta_weight) * self.memories["portfolio_value"][-1] // self.data[self.price_colname].values.tolist()[index] )*
                #     self.data[self.price_colname].values.tolist()[index] * (1-self.transaction_fee_percent))
                #print("(sell) price to get now: ")
                #print(self.data[self.price_colname].values.tolist()[index] * n_assets_to_sell * (1 - self.transaction_fee_percent))
                #print("delta_weight")
                #print(delta_weight)
                #print("portfolio_value all:")
                #print(self.memories["portfolio_value"])
                #print("portfolio_value last:")
                #print(self.memories["portfolio_value"][-1])
                #print("n_assets_to_sell")
                #print(n_assets_to_sell)
            exercised_actions[index] = -n_assets_to_sell
            # update cash balance; cash new = cash old + price * n_assets sold*(1 - transaction cost)
            self.memories["cash_value"][-1] += self.data[self.price_colname].values.tolist()[index] * \
                                     n_assets_to_sell * (1 - self.transaction_fee_percent)
            self.memories["number_asset_holdings"][-1][index] -= n_assets_to_sell
            # update transaction cost; cost + new cost(price * n_assets_sold * transaction fee)
            self.cost += self.data[self.price_colname].values.tolist()[index] * \
                         n_assets_to_sell * self.transaction_fee_percent
            # update sell trades counter
            self.sell_trades += 1

            #if self.mode == "train" and self.steps_counter <= 3:
            #    print("\nmoney to desinvest in this stock (after tc):")
            #    print(money_to_desinvest)
            #    print("money freed (incl. tc):")
            #    print(self.data[self.price_colname].values.tolist()[index] * \
            #                         n_assets_to_sell * (1 - self.transaction_fee_percent))
            #    print("current cash:")
            #    print(self.state["cash"][0])

            return exercised_actions # here, exercised actions = number of stocks sold

        ### PERFORM SELLING USING FUNCTIONS DEFINED ABOVE
        if self.step_version == "paper":
            exercised_actions = _sell_paper(index, action, exercised_actions)
        elif self.step_version == "newNoShort":
            exercised_actions = _sell_newNoShort(index, action, exercised_actions)
        return exercised_actions

    def _buy_stock(self, index: int, action, exercised_actions) -> list:

        def _buy_paper(index, action, exercised_actions) -> list:
            """
            We buy assets based on our policy actions given, under budget constraints.
            We cannot borrow in this setting, hence we can only buy as many assets as we can afford.
            We cannot buy fraction of assets in this setting.
            """
            # max_n_assets_to_buy = cash balance / stock price, rounded to the floor (lowest integer)
            max_n_assets_to_buy = self.state["cash"][0] // self.data[self.price_colname].values.tolist()[index]
            # using the policy actions and budget constraints, get the actually exercisable action
            exercised_action = min(max_n_assets_to_buy, round(action))
            exercised_actions[index] = exercised_action
            # update cash position: old cash - new cash(price * action * (1-cost))
            self.state["cash"][0] -= self.data[self.price_colname].values.tolist()[index] * \
                                     exercised_action * (1 + self.transaction_fee_percent)
            # update asset holdings for the current asset: old holdings + action
            self.state["n_asset_holdings"][index] += exercised_action
            # update transaction cost counter: price * action * cost
            self.cost += self.data[self.price_colname].values.tolist()[index] * \
                         exercised_action * self.transaction_fee_percent
            # update buy trades counter
            self.buy_trades += 1
            return exercised_actions

        def _buy_new(index, action, exercised_actions) -> list:
            """
            We buy assets based on our policy actions given, under budget constraints.
            We cannot borrow in this setting, hence we can only buy as many assets as we can afford.
            We cannot buy fraction of assets in this setting.
            """
            delta_weight = action # (+), as a fraction of total portfolio value
            # for buy action, we need to use the delta_weight corrected for the transaction cost, because the actions need to be self-financing
            delta_weight_tc = delta_weight * (1-self.transaction_fee_percent) * (1 - self.rebalance_penalty)
            # then convert the delta_weight (corrected for transaction cost) in the money to invest in stocks of this stock
            # (note: we need to use the weights corrected for the transaction fee, because if we
            # e.g. invest 1000 USD in stock X, and pay a transaction fee of 1, then we actually only invest 999 of money in the stock
            # (unless we pay 1001, but then it is not self-financing anymore)
            money_to_invest = delta_weight_tc * self.memories["portfolio_value"][-1]
            # convert in max. assets to buy. // takes the floor (rounds down)
            n_assets_to_buy = money_to_invest // self.data[self.price_colname].values.tolist()[index]

            exercised_actions[index] = n_assets_to_buy
            # update cash position: old cash - new cash(price * action * (1-cost))
            self.memories["cash_value"][-1] -= self.data[self.price_colname].values.tolist()[index] * \
                                     n_assets_to_buy * (1 + self.transaction_fee_percent)

            #print("(buy) price to pay if delta-weight used: ")
            #print((delta_weight * self.memories["portfolio_value"][-1] // self.data[self.price_colname].values.tolist()[index] )*
            #      self.data[self.price_colname].values.tolist()[index] * (1+self.transaction_fee_percent))
            #print("(buy) price to pay now: ")
            #print(self.data[self.price_colname].values.tolist()[index] * n_assets_to_buy * (1 + self.transaction_fee_percent))
            # update asset holdings for the current asset: old holdings + action
            self.memories["number_asset_holdings"][-1][index] += n_assets_to_buy
            # update transaction cost counter: price * action * cost
            self.cost += self.data[self.price_colname].values.tolist()[index] * \
                         n_assets_to_buy * self.transaction_fee_percent
            # update buy trades counter
            self.buy_trades += 1

            #if self.mode == "train" and self.steps_counter <= 3:
            #    print("\nmoney to invest in this stock (after tc):")
            #    print(money_to_invest)
            #    print("money invested (incl. tc):")
            #    print(self.data[self.price_colname].values.tolist()[index] * \
            #                         n_assets_to_buy * (1 + self.transaction_fee_percent))
            #    print("current cash:")
            #    print(self.state["cash"][0])

            return exercised_actions

        ### PERFORM BUYING
        if self.step_version == "paper":
            exercised_actions = _buy_paper(index, action, exercised_actions)
        elif self.step_version == "newNoShort":
            exercised_actions = _buy_new(index, action, exercised_actions)
        return exercised_actions

    def reset(self, day: int=None, initial: str="") -> np.array:
        """
        Note: I included day and initial as paremeters here, which is not tootally in line with gym.Env,
        (at least not intended by them), but I need it for having more control over the env during training
        in my custom model
        """
        # We reset the environment to the initial values

        # Reset counter goes up by 1 (because it counts how often we have reset our env since the moment we
        # created the env instance
        self.reset_counter += 1
        # set the steps counter to 0 (because if we reset the env, it starts at the beginning of the episode)
        self.steps_counter = 0
        # set terminal to False, because we start with the initial state of the episode when we reset the env
        self.terminal_state = False
        # this is a hackaround for some debugging, basically if I pass "day" into the reset function,
        # it takes this value for updating the self.day variable, else it takes the self.firstday variable we have defined in the init
        if day is not None:
            self.day = day
        else:
            self.day = self.firstday

        # some self.logger to the self.logger file, for analysis / check if everything is correct
        self.logger.info(f"({self.mode}) reset env, initial episode = {self.initial}, day = {self.day}")

        # get the first observation /initial state from the provided train / validation / test dataset
        self.data = self.df.loc[self.day, :]
        # get datadate from initial state from the provided train / validation / test dataset
        self.datadate = list(self.data["datadate"])[0]
        # overwrite datadates in memories dictionary with initial datadate
        self.memories["datadates"] = self.datadate
        # initialize reward as 0; it is updated after each step (after new day sampled)
        self.reward = 0
        # initialize cost (transaction cost); update after each trade
        self.cost = 0
        # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
        self.buy_trades = 0
        self.sell_trades = 0

        # NOW:
        # we make a difference between whether we are in the first episode of our expanding window training
        # setup or not in the first episode

        # If we are in the first episode (initial, self.initial = True):
        # we initialize all variables "normal", the same way as in the __init__() function
        if self.initial or initial:
            self.logger.info(f"({self.mode} - initial episode, reset env to starting state.")
            # initialize current state
            self.current_cash_balance = self.initial_cash_balance
            self.current_n_asset_holdings = [0] * self.assets_dim
            self.current_asset_equity_weights_startofday = [0] * self.assets_dim
            self.current_cash_weight = 1
            self.current_all_weights_startofday = [0] * self.assets_dim + [self.current_cash_weight] # cash has 100% weight in the beginning
            self.observed_asset_prices_list = self.data[self.price_colname].values.tolist()

            if self.step_version == "paper":
                self.state = {"cash": [self.current_cash_balance],
                              "n_asset_holdings": self.current_n_asset_holdings}
            elif self.step_version == "newNoShort":
                self.state = {"cash_w": [self.current_cash_weight],
                              "asset_w": self.current_all_weights_startofday[:-1]}

            if self.iteration == 1 and self.steps_counter == 0:
                self.asset_names = self.data["tic"].unique()
                if self.step_version == "paper":
                    self.state_header = ["cash"] + [s + "_n_holdings" for s in self.asset_names]
                elif self.step_version == "newNoShort":
                    self.state_header = ["cash_w"] + [s + "_w" for s in self.asset_names]

            for feature in self.features_list:
                self.state.update({feature: self.data[feature].values.tolist()})
                # if we are in the first iteration (=episode) and at the first step,
                # we add feature names with a suffix to th state header from before
                if self.iteration == 1 and self.steps_counter == 0:
                    suffix = "_" + feature
                    self.state_header += [s + suffix for s in self.asset_names]

            for feature in self.single_features_list:
                # for features which are not attached to a stock, like the vix (vixDiv100),
                # we don't want them to appear n_asset times in the state
                self.state.update({feature: [self.data[feature].values[0]]})
                if self.iteration == 1 and self.steps_counter == 0:
                    self.state_header += ["vixDiv100"]

            self.lstm_state = dict((k, self.state[k]) for k in self.lstm_features_list if k in self.state)
            self.lstm_state_flattened = np.asarray(list(chain(*list(self.lstm_state.values()))))
            # create flattened state_flattened
            self.state_flattened = np.asarray(list(chain(*list(self.state.values()))))

            ##### INITIALIZE MEMORIES / MEMORY TRACKERS
            # update after each step (after new day sampled)
            self.memories = {"datadates": [self.datadate],
                             "cash_value": [self.initial_cash_balance],
                             "portfolio_value": [self.initial_cash_balance],
                             "rewards": [self.reward],
                             "policy_actions": [],
                             "policy_actions_trans": [],
                             "exercised_actions": [],
                             "transaction_cost": [],
                             "number_asset_holdings": [self.current_n_asset_holdings],
                             "asset_equity_weights": [self.current_asset_equity_weights_startofday],
                             "all_weights_cashAtEnd": [self.current_all_weights_startofday],
                             "sell_trades": [],
                             "buy_trades": [],
                             "state_memory": [self.state_flattened]}
            if self.iteration == 1 and self.steps_counter == 0:
                self.memories.update({"state_header": self.state_header})

        # If we are NOT in the first episode:
        # for the test set, we want to provide it with the last state at which it stopped in the last episode
        # before the window got extended by settings.ROLL_WINDOW.
        # (note; episode here means training episode; but for each training episode, there is one test episode,
        # so if we are not in the first episode, that means that there was already a test period and we want to use the
        # last state (e.g. asset holdings, cash position)
        # of this previous testing period to continue testing from there on (like we would when we would do training)
        else:
            self.logger.info(f"({self.mode} - not initial episode, reset env with last state of previous episode as starting state.")
            # if any subsequent episode, not initial, we pass the environment the last state
            # including the latest asset holdings / weights, so that the algorithm doesn't have to start from scratch
            # initialize state based on previous state (= last state)
            # basically, the terminal state of the previous episode is going to be the starting state of the current episode,
            # because we did not yet do an action for this state in the last episode
            # Note: the previous state is passed at the initialization of the environment (see __init__() function)
            previous_asset_prices = self.previous_asset_price

            if self.step_version == "paper":
                self.current_n_asset_holdings = self.previous_state["n_asset_holdings"]
                self.current_cash_balance = self.previous_state["cash"][0]
                starting_portfolio_value = self.current_cash_balance + \
                                           sum(np.array(self.current_n_asset_holdings) * np.array(
                                               previous_asset_prices))
            if self.step_version == "newNoShort":
                #print("previous state keys:")
                #print(self.previous_state.keys())
                #print("previous state PF value:")
                #print(self.previous_state["portfolio_value"])
                starting_portfolio_value = self.previous_state["portfolio_value"][0]
                #print("starting_portfolio_value using previous state")
                #print(starting_portfolio_value)
                self.current_n_asset_holdings = np.array(self.previous_state["asset_w"]) * starting_portfolio_value / np.array(previous_asset_prices)
                #print("current_n_asset_holdings")
                #print(self.current_n_asset_holdings)
                self.current_cash_balance = self.previous_state["cash_w"][0] * starting_portfolio_value

            # weights of each asset in terms of equity pf value = number of assets held * asset prices / equity portfolio value
            self.current_asset_equity_weights_startofday = np.array(self.current_n_asset_holdings) * np.array(previous_asset_prices) / (starting_portfolio_value-self.current_cash_balance)
            # eights of each asset and of cash in terms of total portfolio value = number of assets held * asset prices / total pf value, then append cash weight at the end of the list
            self.current_all_weights_startofday = list(np.array(self.current_n_asset_holdings) * np.array(previous_asset_prices) / starting_portfolio_value) + [self.current_cash_balance / starting_portfolio_value]

            # update state dict
            # if we use the paper-version of step calculation, we update the state vector as number of asset holdings
            if self.step_version == "paper":
                self.state = {"cash": [self.current_cash_balance],
                              "n_asset_holdings": self.current_n_asset_holdings}

            elif self.step_version == "newNoShort":
                # here we already use cash weights, so no need for cash scaling
                self.state = {"cash_w": [self.previous_state["cash_w"][0]],
                              "asset_w": self.current_all_weights_startofday[:-1]}

            for feature in self.features_list:
                self.state.update({feature: self.data[feature].values.tolist()})
                # if we are in the first iteration (=episode) and at the first step,
                # we add feature names with a suffix to th state header from before

            for feature in self.single_features_list:
                # for features which are not attached to a stock, like the vix (vixDiv100),
                # we don't want them to appear n_asset times in the state
                self.state.update({feature: [self.data[feature].values[0]]})

            self.lstm_state = dict((k, self.state[k]) for k in self.lstm_features_list if k in self.state)
            self.lstm_state_flattened = np.asarray(list(chain(*list(self.lstm_state.values()))))
            # create flattened state
            self.state_flattened = np.asarray(list(chain(*list(self.state.values()))))


            ##### INITIALIZE MEMORIES / MEMORY TRACKERS
            self.memories = {"datadates": [self.datadate],
                             "cash_value": [self.current_cash_balance],
                             "portfolio_value": [starting_portfolio_value],
                             "rewards": [self.reward],
                             "policy_actions": [],
                             "policy_actions_trans": [],
                             "exercised_actions": [],
                             "transaction_cost": [],
                             "number_asset_holdings": [self.current_n_asset_holdings],
                             "asset_equity_weights": [self.current_asset_equity_weights_startofday],
                             "all_weights_cashAtEnd": [self.current_all_weights_startofday],
                             "sell_trades": [],
                             "buy_trades": [],
                             "state_memory": [self.state_flattened]}
        return self.state_flattened, self.lstm_state_flattened

    def return_reset_counter(self) -> int:
        return self.reset_counter

    def return_steps_counter(self) -> int:
        return self.steps_counter

    def render(self, mode="human"):
        # here we return:
        # the state (flattened) as list / numpy array
        # the state as dictionary with the last portfolio value appended as well because if we use the custom version,
        #   we need to pass the portfolio value as well because we don't store the cash value ans number of assets held
        #   in the state vector and hence could not compute the portfolio value
        # the observed last asset prices list (for debug)
        # the reset counter (only needed for debug)
        # the final state counter (only needed for debug)
        # the steps counter (only needed for debug)
        #print("last pf value (render):")
        #print(self.memories["portfolio_value"][-1])
        self.state.update({"portfolio_value": [self.memories["portfolio_value"][-1]]})
        #print("self.state (dict)")
        #print(self.state)
        return self.state_flattened, self.lstm_state_flattened, self.state, self.observed_asset_prices_list, \
               self.reset_counter, self.final_state_counter, self.steps_counter