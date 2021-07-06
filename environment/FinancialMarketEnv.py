import numpy as np
import pandas as pd
from itertools import chain
import logging
import gym
from gym import spaces

# import own libraries
try:
    from config.config import *
except:
    from config import *

class FinancialMarketEnv(gym.Env):
    """A stock trading environment for OpenAI gym
    see also: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
    complete guide on how to create a custom env. with OpenAI gym :
              https://github.com/openai/gym/blob/master/docs/creating-environments.md
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 # whole train / validation or test dataset
                 df: pd.DataFrame,
                 # number of assets
                 assets_dim: int,
                 # how long the observation vector should be for one day;
                 # n_stocks*n_features_per_stock+ n_stocks(for saving asset holdings) + n_other_features (if any) + 1 (for cash position)
                 shape_observation_space: int,
                 # where results hsoud be saved
                 results_dir: str,
                 # list of features names to be used
                 features_list: list = data_settings.FEATURES_LIST,
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
                 iteration: str = "",
                 # this is mutiplied by the actions given by the policy, which are usually between -10, 10,
                 # in order to get the number of assets to buy
                 hmax_normalize: int = env_params.HMAX_NORMALIZE,
                 # this is the starting cash balance. In this work, we are not allowed to short or to buy more than we can afford
                 initial_cash_balance: float = env_params.INITIAL_CASH_BALANCE,
                 # this is a flat rate transaction cost which is multiplied by the trading volume
                 transaction_fee_percent: float = env_params.TRANSACTION_FEE_PERCENT,
                 # if rewards are in absolute portfolio change values, they need to be scaled down a bit because they
                 # can be quite large, like 2000, this is better for convergence
                 reward_scaling: float = env_params.REWARD_SCALING,
                 # whether we are in the initial episode or not.
                 # if we are not in the initial episode, then for the test set we pass the previous state to it.
                 # for the train set, it depends whether we want to retrain or continue training with the saved model
                 initial: bool = True,
                 # previous state, if available
                 previous_state: list = [],
                 # previous asset price list, if available
                 previous_asset_price: list = [],
                 # name of the price column, here "datadate"
                 price_colname: str = data_settings.MAIN_PRICE_COLUMN,
                 # a counter for how often the env was reset, for analysis / debugging
                 reset_counter: int = 0,
                 # a counter of how often the agent reached the final state (= end of the provied train / validation / test dataset)
                 # also used for analysis / debugging
                 final_state_counter: int = 0,
                 # counter of how many steps were taken in one episode, used for saving results and for analysis / debugging
                 steps_counter: int = 0,
                 # whether we want to save results or not (default True, but for debugging sometimes False)
                 save_results=True):
        # we call the init function in the class gym.Env
        super().__init__()
        """
        @param df: pd.DataFrame(), sorted by date, then ticker, index column is the datadate factorized
                   (split_by_date function)
        ...
        """
        ##### INPUT VARIABLES TO THE CLASS
        self.reset_counter = reset_counter
        self.final_state_counter = final_state_counter
        self.steps_counter = steps_counter

        self.mode = mode
        self.df = df.copy()
        self.features_list = features_list
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

        self.price_colname = price_colname
        self.results_dir = results_dir
        self.save_results = save_results

        ##### CREATING ADDITIONAL VARIABLES
        # action_space normalization and shape is assets_dim
        self.data = self.df.loc[self.day, :]  # includes all tickers, hence >1 line for >1 assets
        self.datadate = list(self.data[data_settings.DATE_COLUMN])[0]  # take first element of list of identical dates

        # we change the action space; instead from -1 to 1, it will go from 0 to 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.assets_dim,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.shape_observation_space,))
        # todo: changed from 0:np.inf to -np.inf:np.inf

        ##### INITIALIZING VARIABLES
        # set terminal state to False at initialization
        # Note: this is the same variable as "done" or "dones", the so-called "mask" in reinforcement learning, just renamed
        self.terminal_state = False

        # initializing state for current day:
        # current cash balance is the initial cash balance
        self.current_cash_balance = self.initial_cash_balance
        # number of asset holdings and asset qeuity weights are each a vector of zeroes with the same length as number of assets (one place for each asset)
        self.current_n_asset_holdings = [0] * self.assets_dim
        # weights are a bit "special"; because they change twice in a state transition;
        # first, when we rebalance our portfolio, we change our weights but with the old asset prices
        # second, when the we get the new day / state and observe the new asset prices, the (money-) weights o our assets change again
        # here, we only record the (money-)weights of each asset at the beginning of the day, meaning: after both changes mentioned above
        # so we start with weights of 0; the next state will be n_asset_holdings*new_asset_price / (equity portfolio value)
        self.current_asset_equity_weights_startofday = [0] * self.assets_dim
        # then it is also interesting to track how the weights of all stocks change compared to the whole pf value (incl, cash)
        # and how much weight cash has in the portfolio, hence we create a vector of zeroes of length n_assets + 1 (for cash)
        # the last list entry will be for cash
        self.current_all_weights_startofday = [0] * self.assets_dim + [1]
        # in order to have it simpler to query, I created a dictionary for the state, where all the things to be saved
        # are put in there and accessible by "keyname"
        self.state = {"cash": [self.current_cash_balance],
                      "n_asset_holdings": self.current_n_asset_holdings}
        # if we are at the first time step of the episode, we get the asset names (don't need to do this at every step, but we could)
        if self.steps_counter == 0:
            self.asset_names = df[data_settings.ASSET_NAME_COLUMN].unique()
        # if we are at the first iteration (= first episode), and in the first step,
        # (we start counting episodes from 1, steps from 0, just because it was more practical for printing the number of episodes)
        # we create the state header for later being able to save all the states in one dataframe together with the header
        if self.iteration == 1 and self.steps_counter == 0:
            self.state_header = ["cash"] + [s + "_n_holdings" for s in self.asset_names]
        # for each feature in the features list we passed in the init, we update the dictionary;
        # we update the state dict with the features
        for feature in self.features_list:
            self.state.update({feature: self.data[feature].values.tolist()})
            # if we are in the first iteration (=episode) and at the first step,
            # we add feature names with a sufix to th state header from before
            if self.iteration == 1 and self.steps_counter == 0:
                suffix = "_" + feature
                self.state_header += [s + suffix for s in self.asset_names]
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
                         "exercised_actions": [],
                         "transaction_cost": [],
                         "number_asset_holdings": [self.current_n_asset_holdings],
                         "asset_equity_weights": [self.current_asset_equity_weights_startofday], # asset weights within equity part (= n_asset_holdings/total_asset_holdings)
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
        transitio from one state to the other state, using actions provided by the RL agent.

        First, we chneck if we are in the terminal state of the episode (done=True) or not.
            If we are in the terminal state, we don't take any actions anymore, but just return the final state and reward,
                    and save the results (memory and state vector) to csv files
            If we are not in the terminal state, we actually take a ste using the actions provided by the agent,
                    then the environment samples and returns the next state and a reward

        methods:
        _save_results()     : saving results to csv at the end of the episode.
        _terminate()        : this function is called if we are in the terminal state of the episode;
                              we then append the last data to the memory dictionary and call the _save_results() function
                              (if save_results == True).
                              Output:
                              self.state_flattened: np.array, self.reward: float, self.terminal_state: bool, {}
        _calculate_reward   : Inputs: end_portfolio_value: float=None, begin_portfolio_value: float=None
                              This function calculates the reward, depending on the calculation method specified
                              in config.py under settings.REWARD_MEASURE.
        _sell_stock()       : selling stocks
        _buy_stock()        : buying stocks

        @param actions: np.array of actions, provided by the agent
        @return: self.state: np.array, self.reward: float, self.terminal_state: bool, {}
        """
        def _save_results() -> None:
            # SAVING MEMORIES TO CSV
            # save dates vector alone to results (for debugging, analysis, plotting...)
            dates = pd.DataFrame({"datadate": self.memories["datadates"]})
            dates.to_csv(os.path.join(self.results_dir, "datadates",
                                        f"datadates"
                                        f"_{self.mode}"
                                        f"_{self.model_name}"
                                        f"_ep{self.iteration}" # episode
                                        f"_totalSteps_{self.steps_counter}"
                                        f"_finalStateCounter_{self.final_state_counter}"
                                        f".csv"))
            # if we are at the last step of the first episode, save header (stays the same for whole run,
            # # so no need to save again every time)
            if self.iteration == 1:
                pd.DataFrame(self.memories["state_header"]).\
                        to_csv(os.path.join(self.results_dir, "state_memory", "state_header.csv"))
            # then, for each data entry in the memories dictionary, we save together with the corresponding dates to csv
            for key in list(self.memories.keys()):
                # create pandas df for each key (except for datadate and state_header, because we already saved them before
                if key not in ["datadate", "state_header"]:
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
                if key not in ["datadate", "state_header"]:
                    pd.concat([dates, keydf], axis=1).to_csv(os.path.join(self.results_dir, key,
                                                                      f"{key}"
                                                                      f"_{self.mode}"
                                                                      f"_{self.model_name}"
                                                                      f"_ep{self.iteration}"
                                                                      f"_totalSteps_{self.steps_counter}"
                                                                      f"_finalStateCounter_{self.final_state_counter}.csv"))
            return None

        def _terminate() -> list:
            self.final_state_counter += 1
            self.memories["exercised_actions"].append([0] * self.assets_dim)  # because no actions exercised in this last date anymore
            self.memories["transaction_cost"].append(0)  # since no transaction on this day, no transaction costcost
            self.memories["sell_trades"].append(0)
            self.memories["buy_trades"].append(0)
            # SAVING MEMORIES TO CSV
            if self.save_results == True:
                _save_results()
            return [self.state_flattened, self.reward, self.terminal_state, {}]

        def _calculate_reward(end_portfolio_value: float=None, begin_portfolio_value: float=None):
            if settings.REWARD_MEASURE == "addPFVal":
                self.reward = end_portfolio_value - begin_portfolio_value
                # apply reward scaling
                self.reward = self.reward * self.reward_scaling
            elif settings.REWARD_MEASURE == "SR7": # 7 day sharpe ratio (non-annualized)
                portfolio_return_daily = pd.DataFrame(self.memories["portfolio_value"]).pct_change(1)
                sharpe_ratio_7days = portfolio_return_daily[-7:].mean() / portfolio_return_daily[-7:].std()
                self.reward = sharpe_ratio_7days.values.tolist()[0]
            elif settings.REWARD_MEASURE == "logU": # log Utility(new F value / old PF value) as proposed by Neuneier 1997
                self.reward = np.log(end_portfolio_value / begin_portfolio_value)
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
            begin_cash_value = self.state["cash"][0]
            begin_n_asset_holdings = self.state["n_asset_holdings"]
            begin_asset_prices = self.data[self.price_colname].values.tolist()
            begin_portfolio_value = begin_cash_value + sum(np.array(begin_asset_prices) * np.array(begin_n_asset_holdings))

            ################################################################################################
            ### TAKE A STEP,
            ### (WITHIN THE CURRENT STATE, BEFORE THE ENVIRONMENT SAMPLES A NEW STATE)
            ################################################################################################
            # we sort the actions such that they are in this order [-large, ..., +large]
            # and we get the "indizes" of at which position in the original actions array the sorted actions were
            argsort_actions = np.argsort(actions)
            # get assets to be sold (if action -)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            # get assets to be bought (if action +)
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            # create empty list which will be filled with sell / buy actions for each asset
            exercised_actions = [0] * self.assets_dim

            # TODO: Note: this iterates index by index (stock by stock) and sells / buys stocks in consecutive order.
            # todo: for buy: can it happen that one stock uses up all recources an dthe others then cannot be bought anymore, in the extreme case?
            # todo: is there a way to buy stocks based on their *fraction* in the portfolio, instead based on number of stocks? since one
            # todo: cannot buy /sell more than 100 stocks and the cash for buying is also limited
            for index in sell_index:  # index starting at 0
                exercised_actions = self._sell_stock(index, actions[index], exercised_actions)
            for index in buy_index:
                exercised_actions = self._buy_stock(index, actions[index], exercised_actions)

            ### UPDATE VALUES AFTER ACTION TAKEN (WITHIN THE CURRENT STATE, BEFORE THE ENVIRONMENT SAMPLES A NEW STATE)
            # after we took an action, what changes immediately and independently of the next state we will get from
            # the environment, are the number of asset holdings, the new cash balance and
            self.current_cash_balance = self.state["cash"][0]
            self.current_n_asset_holdings = list(self.state["n_asset_holdings"])

            # append new data after taking the actions
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
            self.datadate = list(self.data[data_settings.DATE_COLUMN])[0]
            self.memories["datadates"].append(self.datadate)
            # get new asset prices
            self.observed_asset_prices_list = self.data[self.price_colname].values.tolist()
            # create new state dictionary; append current cash position and current asset holdings (after taking a step)
            self.state = {"cash": [self.current_cash_balance],
                          "n_asset_holdings": self.current_n_asset_holdings}
            # for each provided feature, append the feature value for the new day to the state dictionary
            for feature in self.features_list:
                self.state.update({feature: self.data[feature].values.tolist()})
            # create flattened state (because we need to pass a np.array to the model which then converts it to a tensor,
            # cannot use a dictionary; but the dictionary is practical for querying and not loosing the overview of what is where)
            self.state_flattened = np.asarray(list(chain(*list(self.state.values()))))

            # get asset prices of new day
            current_asset_prices = self.data[self.price_colname].values.tolist()
            # calculate current portfolio value, after taking a step (which changed the cash balance and number of asset holdings / pf weights)
            # and after getting a new day sample from the environment (with new asset prices)
            current_portfolio_value = self.current_cash_balance + sum(np.array(self.current_n_asset_holdings) *
                                                                  np.array(current_asset_prices))

            # weights of each asset in terms of equity pf value = number of assets held * asset prices / equity portfolio value
            self.current_asset_equity_weights_startofday = np.array(self.current_n_asset_holdings) * np.array(current_asset_prices) / (current_portfolio_value-self.current_cash_balance)
            # eights of each asset and of cash in terms of total portfolio value = number of assets held * asset prices / total pf value, then append cash weight at the end of the list
            self.current_all_weights_startofday = list(np.array(self.current_n_asset_holdings) * np.array(current_asset_prices) / current_portfolio_value) + [self.current_cash_balance / current_portfolio_value]


            ### CALCULATE THE REWARD, FOLLOWING THE PREVIOUS STATE, ACTION PAIR
            _calculate_reward(end_portfolio_value=current_portfolio_value,
                              begin_portfolio_value=begin_portfolio_value)
            #self.reward = end_portfolio_value - begin_portfolio_value

            # Append rewards, new portfolio value and the new complete state vector to the memory dict
            self.memories["rewards"].append(self.reward)
            self.memories["portfolio_value"].append(current_portfolio_value)
            #self.memories["cash_value"].append(self.state["cash"][0]) # todo: rm, already appended above after actions
            self.memories["state_memory"].append(self.state_flattened)
            self.memories["asset_equity_weights"].append(self.current_asset_equity_weights_startofday)
            self.memories["all_weights_cashAtEnd"].append(self.current_all_weights_startofday)

            ### RESET SOME COUNTERS
            # we want to get the transaction cost, sell trades and buy trades accumulated daily only (on a per step basis)
            # not accumulated over multiple days / steps
            self.cost = 0
            self.sell_trades = 0
            self.buy_trades = 0
            return [self.state_flattened, self.reward, self.terminal_state, {}]


        ####################################
        #    MAIN CODE FOR STEP FUNCTION   #
        ####################################
        # if the sampled day index is larger or equal to the last index available in the data set provided,
        # we have reached the end of the episode.
        # Note: there are many ways to define an end of the episode, like when a certain reward / penalty is reached,
        # or we could also define fixed intervals for eth episodes, like 30 days etc., but here it makes sense to treat this
        # as if the episode is as long as the training data available, because theoretically it is a continuous problem,
        # where one invests over multiple years, theoretically "with no fixed end"
        self.terminal_state = self.day >= self.df.index.unique()[-1] # :bool

        # So the actions from the model come out in the range of, like, -2, ..., +2 plus/minus
        # and then they are reshaped to the action space we have defined in the environment (reshaping is done in the ppo algorithm)
        # so that there will be no out-of-bound errors
        # and then here in this setup, the actions are multiplied by hmax, so that we get "number of stocks to buy or sell"
        actions = actions * self.hmax_normalize
        # save policy actions (actions) to memories dict (these are the actions given by the agent, not th actual actions taken)
        self.memories["policy_actions"].append(actions)

        ##### IF WE ARE IN THE TERMINAL STATE #####
        if self.terminal_state:
            # we call the function to terminate the episode, no step will be taken anymore
            return _terminate()
        ##### IF WE ARE NOT YET IN THE TERMINAL STATE #####
        else:
            # we take a step using the policy actions
            return _take_step(actions=actions)

    def _sell_stock(self, index, action, exercised_actions) -> list:

        def _sell_normal(index, action, exercised_actions) -> list:
            """
            If we hold assets, we can sell them based on our policy actions, but under short-selling constraints.
            Hence, we cannot sell more assets than we own.
            If we don't hold any assets, we pass, since we cannot short-sell.
            """
            # if we hold assets of the stock (given by the index), we can sell
            if self.state["n_asset_holdings"][index] > 0:
                # todo: document; I have changed everywhere into round() because else actions (n.stocks to buy would be floats!)
                # todo: changed how state is constructed etc.
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

        ### PERFORM SELLING USING FUNCTIONS DEFINED ABOVE
        exercised_actions = _sell_normal(index, action, exercised_actions)
        return exercised_actions

    def _buy_stock(self, index: int, action, exercised_actions) -> list:

        def _buy_normal(index, action, exercised_actions) -> list:
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

        ### PERFORM BUYING
        exercised_actions = _buy_normal(index, action, exercised_actions)
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

        # some logging to the logging file, for analysis / check if everything is correct
        logging.info(f"({self.mode}) reset env, initial episode = {self.initial}, day = {self.day}")

        # get the first observation /initial state from the provided train / validation / test dataset
        self.data = self.df.loc[self.day, :]
        # get datadate from initial state from the provided train / validation / test dataset
        self.datadate = list(self.data[data_settings.DATE_COLUMN])[0]
        # overwrite datadates in memories dictionary with initial datadate
        self.memories["datadates"] = self.datadate
        # initialize reward as 0; it is updated after each step (after new day sampled)
        self.reward = 0
        # initialize cost (transaction cost); update after each trade
        self.cost = 0
        # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
        self.buy_trades = 0
        self.sell_trades = 0

        # todo: rm
        # print("firstday: ", self.firstday)
        # print("day: ", self.day)
        #print("(env reset) day: ", day)
        #print("(env reset) self.day: ", self.day)
        #print("(env reset) self.data: ", self.data)
        # if we are in the first episode, we do not have a previous state

        # NOW:
        # we make a difference between whether we are in the first episode of our expanding window training
        # setup or not in the first episode

        # If we are in the first episode (initial, self.initial = True):
        # we initialize all variables "normal", the same way as in the __init__() function
        if self.initial or initial:
            # initialize current state
            self.current_cash_balance = self.initial_cash_balance
            self.current_n_asset_holdings = [0] * self.assets_dim
            self.current_asset_equity_weights_startofday = [0] * self.assets_dim
            self.current_all_weights_startofday = [0] * self.assets_dim + [1] # cash has 100% wieght in the beginning

            self.observed_asset_prices_list = self.data[self.price_colname].values.tolist()
            self.state = {"cash": [self.current_cash_balance],
                          "n_asset_holdings": self.current_n_asset_holdings}
            if self.iteration == 1 and self.steps_counter == 0:
                self.asset_names = self.data[data_settings.ASSET_NAME_COLUMN].unique()
                self.state_header = ["cash"] + [s + "_n_holdings" for s in self.asset_names]
            for feature in self.features_list:
                self.state.update({feature: self.data[feature].values.tolist()})
                if self.iteration == 1 and self.steps_counter == 0:
                    suffix = "_" + feature
                    self.state_header += [s + suffix for s in self.asset_names]
            # create flattened state_flattened
            self.state_flattened = np.asarray(list(chain(*list(self.state.values()))))

            ##### INITIALIZE MEMORIES / MEMORY TRACKERS
            # update after each step (after new day sampled)
            self.memories = {"datadates": [self.datadate],
                             "cash_value": [self.initial_cash_balance],
                             "portfolio_value": [self.initial_cash_balance],
                             "rewards": [self.reward],
                             "policy_actions": [],
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
            print("not initial episode")
            # if any subsequent episode, not initial, we pass the environment the last state
            # including the latest asset holdings / weights, so that the algorithm doesn't have to start from scratch
            # initialize state based on previous state (= last state)
            # basically, the terminal state of the previous episode is going to be the starting state of the current episode,
            # because we did not yet do an action for this state in the last episode
            # Note: the previous state is passed at the initialization of the environment (see __init__() function)

            self.current_n_asset_holdings = self.previous_state["n_asset_holdings"]
            self.current_cash_balance = self.previous_state["cash"][0]
            previous_asset_prices = self.previous_asset_price
            starting_portfolio_value = self.current_cash_balance + \
                                       sum(np.array(self.current_n_asset_holdings) * np.array(previous_asset_prices))
            # weights of each asset in terms of equity pf value = number of assets held * asset prices / equity portfolio value
            self.current_asset_equity_weights_startofday = np.array(self.current_n_asset_holdings) * np.array(previous_asset_prices) / (starting_portfolio_value-self.current_cash_balance)
            # eights of each asset and of cash in terms of total portfolio value = number of assets held * asset prices / total pf value, then append cash weight at the end of the list
            self.current_all_weights_startofday = list(np.array(self.current_n_asset_holdings) * np.array(previous_asset_prices) / starting_portfolio_value) + [self.current_cash_balance / starting_portfolio_value]

            # update state dict
            self.state = {"cash": [self.current_cash_balance],
                          "n_asset_holdings": self.current_n_asset_holdings}
            for feature in self.features_list:
                self.state.update({feature: self.data[feature].values.tolist()})
            # create flattened state
            self.state_flattened = np.asarray(list(chain(*list(self.state.values()))))

            ##### INITIALIZE MEMORIES / MEMORY TRACKERS
            self.memories = {"datadates": [self.datadate],
                             "cash_value": [self.current_cash_balance],
                             "portfolio_value": [starting_portfolio_value],
                             "rewards": [self.reward],
                             "policy_actions": [],
                             "exercised_actions": [],
                             "transaction_cost": [],
                             "number_asset_holdings": [self.current_n_asset_holdings],
                             "asset_equity_weights": [self.current_asset_equity_weights_startofday],
                             "all_weights_cashAtEnd": [self.current_all_weights_startofday],
                             "sell_trades": [],
                             "buy_trades": [],
                             "state_memory": [self.state_flattened]}
        return self.state_flattened

    def return_reset_counter(self) -> int:
        return self.reset_counter

    def return_steps_counter(self) -> int:
        return self.steps_counter

    def render(self, mode="human"):
        return self.state_flattened, self.state, self.observed_asset_prices_list, \
               self.reset_counter, self.final_state_counter, self.steps_counter