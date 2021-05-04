import numpy as np
import pandas as pd
from itertools import chain
import logging
import gym
from gym import spaces
from gym.utils import seeding
from config.config import *


class FinancialMarketEnv(gym.Env):
    """A stock trading environment for OpenAI gym
    see also: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
    complete guide on how to create a custom env. with OpenAI gym :
              https://github.com/openai/gym/blob/master/docs/creating-environments.md
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 assets_dim: int,
                 shape_observation_space: int,
                 results_dir: str,
                 features_list: list = dataprep_settings.FEATURES_LIST,
                 day: int = 0,
                 mode: str = "",  # "validation", "test" =trade
                 model_name: str = "",
                 iteration: str = "",
                 hmax_normalize: int = env_params.HMAX_NORMALIZE,
                 initial_cash_balance: float = env_params.INITIAL_CASH_BALANCE,
                 transaction_fee_percent: float = env_params.TRANSACTION_FEE_PERCENT,
                 reward_scaling: float = env_params.REWARD_SCALING,
                 initial: bool = True,
                 previous_state: list = [],
                 price_colname: str = dataprep_settings.MAIN_PRICE_COLUMN,
                 crisis_measure: str = crisis_settings.CRISIS_MEASURE,
                 crisis_threshold: float = 0,
                 seed=settings.SEED_ENV,
                 run_platform="local",
                 reset_counter: int = 0,
                 final_state_counter: int = 0,
                 steps_counter: int = 0
                 ):
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
        self.df = df
        self.features_list = features_list
        self.firstday = day
        self.day = day
        self.model_name = model_name
        self.iteration = iteration

        self.initial = initial
        self.previous_state = previous_state

        self.crisis_measure = crisis_measure
        self.crisis_threshold = crisis_threshold

        self.hmax_normalize = hmax_normalize
        self.initial_cash_balance = initial_cash_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling
        self.assets_dim = assets_dim
        self.shape_observation_space = shape_observation_space

        self.price_colname = price_colname
        self.results_dir = results_dir

        ##### CREATING ADDITIONAL VARIABLES
        # action_space normalization and shape is assets_dim
        self.data = self.df.loc[self.day, :]  # includes all tickers, hence >1 line for >1 assets
        self.datadate = list(self.data[dataprep_settings.DATE_COLUMN])[
            0]  # take first element of list of identical dates
        # todo: check if action and obserbation space are correct. especially obs.space, why not also -?
        # todo: since we also have negative numbers
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.assets_dim,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.shape_observation_space,))

        ##### INITIALIZING VARIABLES # todo: does this actually need to be done here, or can I do this
        # todo:                              only in reset ?
        # set terminal state to false at initialization
        self.terminal_state = False

        # initializing state for current day
        self.current_cash_balance = self.initial_cash_balance
        self.current_n_asset_holdings = [0] * self.assets_dim
        self.state = {"cash": [self.current_cash_balance],
                      "n_asset_holdings": self.current_n_asset_holdings}
        if self.iteration == 126 or self.iteration == 1:
            asset_names = df[dataprep_settings.ASSET_NAME_COLUMN].unique()
            self.state_header = ["cash"] + [s + "_n_holdings" for s in asset_names]
        for feature in self.features_list:
            self.state.update({feature: self.data[feature].values.tolist()})
            if self.iteration == 126 or self.iteration == 1:
                suffix = "_" + feature
                self.state_header += [s + suffix for s in asset_names]
        # create flattened state_flattened
        self.state_flattened = list(chain(*list(self.state.values())))

        # self.reset() # todo?
        # self._seed(seed) # todo?

        # initialize reward; update after each step (after new day sampled)
        self.reward = 0
        # initialize crisis measure; update after each step (after new day sampled)
        self.crisis = 0
        # initialize transaction cost; update after each trade
        self.cost = 0
        # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
        self.buy_trades = 0
        self.sell_trades = 0
        # initialize number of assets held
        self.n_asset_holdings = self.current_n_asset_holdings

        ##### INITIALIZE MEMORIES / MEMORY TRACKERS
        # note: key names must be the same as for paths.SUBSUBDIR_NAMES,
        # else saving to csv doesn't work (dirs won't match)
        self.memories = {"datadates": [self.datadate],
                         "cash_value": [self.initial_cash_balance],
                         "portfolio_value": [self.initial_cash_balance],
                         # starting value (beginning of the day), before taking a step
                         # "total_assets_value_memory" : [self.initial_cash_balance],
                         "rewards": [self.reward],
                         # reward in first entry is 0 because we can only calculate it after a day has passed (but could be done differnetly, doesn't matter)
                         "policy_actions": [],
                         "exercised_actions": [],
                         "transaction_cost": [],
                         "number_asset_holdings": [self.n_asset_holdings],
                         "sell_trades": [],
                         "buy_trades": [],
                         "state_memory": [self.state_flattened],
                         }
        if self.crisis_measure is not None:
            self.memories.update({"crisis_measures": [self.crisis],
                                  "crisis_thresholds": [self.crisis_threshold],
                                  "crisis_selloff_cease_trading": []})
        if self.iteration == 126 or self.iteration == 1:
            self.memories.update({"state_header": [self.state_header]})

    def step(self, actions):
        """
        Check if we are in the terminal state.
            If yes, return the final state, reward...
            If no, take a step in environment and an action, then load next state and return reward, state...
        This function calls _sell_stock() and _buy_stock()

        @param actions:
        @return: self.state, self.reward, self.terminal_state, {}
        """
        # self.terminal_state = self.day >= len(self.df.index.unique()) - 1
        self.terminal_state = self.day >= self.df.index.unique()[-1]
        # normalized actions need to be multiplied by 100 to get number of stocks to purchase
        actions = actions * self.hmax_normalize
        # save policy actions (actions), update policy memory which is independent of step taken
        self.memories["policy_actions"].append(actions)

        def _save_results() -> None:
            # SAVING MEMORIES TO CSV
            pd.DataFrame({"datadate": self.memories["datadates"]}).\
                to_csv(os.path.join(self.results_dir, "datadates",
                                    f"datadates_{self.mode}_{self.model_name}_"
                                    f"i{self.iteration}_finalStateCounter_{self.final_state_counter}.csv"))
            if self.iteration == 126 or self.iteration == 1: # todo: rm 126 cond
                pd.DataFrame({"state_header": self.memories["state_header"]}
                             ).to_csv(os.path.join(self.results_dir, "state_memory",
                                                   f"state_header_finalStateCounter_{self.final_state_counter}.csv"))
            for key in list(self.memories.keys())[1:-1]:
                pd.DataFrame({"datadate": self.memories["datadates"], key: self.memories[key]}).\
                    to_csv(os.path.join(self.results_dir, key, f"{key}_{self.mode}_{self.model_name}_i{self.iteration}"
                                                        f"_finalStateCounter_{self.final_state_counter}.csv"))
            return None

        ##### IF WE ARE IN THE TERMINAL STATE #####
        if self.terminal_state:
            self.final_state_counter += 1
            self.memories["exercised_actions"].append(
                [0] * self.assets_dim)  # because no actions exercised in this last date anymore
            self.memories["transaction_cost"].append(0)  # since no transaction on this day, no transaction costcost
            self.memories["sell_trades"].append(0)
            self.memories["buy_trades"].append(0)
            # SAVING MEMORIES TO CSV
            _save_results()
            return self.state_flattened, self.reward, self.terminal_state, {}

        ##### IF WE ARE NOT YET IN THE TERMINAL STATE #####
        else:
            self.steps_counter += 1
            if self.crisis_measure is not None:
                if self.crisis >= self.crisis_threshold:
                    # if crisis condition is true, overwrite actions because we want to sell all assets off
                    # so we set our actions to -100 (max. possible number of asset selling)
                    actions = np.array([-self.hmax_normalize] * self.assets_dim)
                    self.memories["crisis_selloff_cease_trading"].append(1)
                elif self.crisis < self.crisis_threshold:
                    self.memories["crisis_selloff_cease_trading"].append(0)

            # begin_cash_value = self.state[0]
            begin_cash_value = self.state["cash"][0]
            # begin_n_asset_holdings = self.state[1:(self.assets_dim + 1)]
            begin_n_asset_holdings = self.state["n_asset_holdings"]
            # begin_asset_prices = self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]
            begin_asset_prices = self.state[self.price_colname]
            begin_portfolio_value = begin_cash_value + \
                                    sum(np.array(begin_asset_prices) * np.array(begin_n_asset_holdings))
            # begin_portfolio_value = self.state[0] + \
            #                       sum(np.array(self.state[1:(self.assets_dim + 1)]) *
            # np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))

            argsort_actions = np.argsort(actions)
            # get assets to be sold (if action -)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            # get assets to be bought (if action +)
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            # create empty list which will be filled with sell / buy acions for each asset
            exercised_actions = [0] * self.assets_dim

            # TODO: Note: this iterates index by index (stock by stock) and sells / buys stocks in consecutive order.
            # todo: for buy: can it happen that one stock uses up all recources an dthe others then cannot be bought anymore, in the extreme case?
            # todo: is there a way to buy stocks based on their *fraction* in the portfolio, instead based on number of stocks? since one
            # todo: cannot buy /sell more than 100 stocks and the cash for buying is also limited
            for index in sell_index:  # index starting at 0
                # print('take sell action'.format(actions[index]))
                exercised_actions = self._sell_stock(index, actions[index], exercised_actions)
            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                exercised_actions = self._buy_stock(index, actions[index], exercised_actions)

            ### UPDATE VALUES AFTER ACTION TAKEN
            # counters to be changed after actions (apart from sell and buy traded,
            # which are change din the sell and buy functions directyl)
            # self.n_asset_holdings = list(self.state[1:(self.assets_dim + 1)])
            self.n_asset_holdings = list(self.state["n_asset_holdings"])
            # memories to be changed after action
            self.memories["exercised_actions"].append(exercised_actions)
            self.memories["sell_trades"].append(self.sell_trades)
            self.memories["buy_trades"].append(self.buy_trades)
            self.memories["transaction_cost"].append(self.cost)
            self.memories["number_asset_holdings"].append(self.n_asset_holdings)

            ### NEW DAY SAMPLING
            # after taking the actions (sell, buy), we update to the next day and get the new data from the dataset
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.datadate = list(self.data[dataprep_settings.DATE_COLUMN])[
                0]  # take first element of list of identical dates
            self.memories["datadates"].append(self.datadate)

            ### ### ### ### ### ### ### ### ##
            ### UPDATE VALUES FOR NEW DAY  ###
            ### ### ### ### ### ### ### ### ##

            # new crisis measure
            if self.crisis_measure is not None:
                self.crisis = self._update_crisis_measure()
                self.memories["crisis_measures"].append(self.crisis)
                self.memories["crisis_thresholds"].append(self.crisis_threshold)
            # update current state # todo: I changes: now it is a dict instead of a list, automized how constructed,
            # todo: and then dict flattened as output for output state.
            self.current_cash_balance = self.state["cash"][0]
            self.current_n_asset_holdings = list(self.state["n_asset_holdings"])
            self.observed_asset_prices_list = self.data[self.price_colname].values.tolist()
            self.state = {"cash": [self.current_cash_balance],
                          "n_asset_holdings": self.current_n_asset_holdings}
            for feature in self.features_list:  # now price included in features list
                self.state.update({feature: self.data[feature].values.tolist()})
            # create flattened state_flattened
            self.state_flattened = list(chain(*list(self.state.values())))

            # final portfolio + cash value after the end of the day with new prices (move up # todo)
            end_cash_value = self.current_cash_balance
            # end_n_asset_holdings = self.state[1:(self.assets_dim + 1)]
            end_n_asset_holdings = self.state["n_asset_holdings"]
            # end_asset_prices = self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]
            end_asset_prices = self.state[self.price_colname]
            end_portfolio_value = end_cash_value + sum(np.array(end_n_asset_holdings) * np.array(end_asset_prices))
            # end_portfolio_value = self.state[0] + \
            #                  sum(np.array(self.state[1:(self.assets_dim + 1)]) *
            #                      np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            self.reward = end_portfolio_value - begin_portfolio_value
            self.memories["rewards"].append(self.reward)
            self.memories["portfolio_value"].append(end_portfolio_value)
            self.memories["cash_value"].append(self.state["cash"][0])
            self.memories["state_memory"].append(self.state_flattened)
            # apply reward scaling
            self.reward = self.reward * self.reward_scaling
            # we want to get the transaction cost for each day, not accumulated over multiple days, same for trades
            self.cost = 0
            self.sell_trades = 0
            self.buy_trades = 0
        return self.state_flattened, self.reward, self.terminal_state, {}

    def _update_crisis_measure(self):
        """
        we update our current crisis measure value, if we have some (if it is not set as None in config.py.
        @return: crisis measure value
                 (e.g. if our crisis measure = turbulence:index; then we reurn current market turbulence)
        """
        if self.crisis_measure == "turbulence":
            self.crisis = self.data[self.crisis_measure].values[0]  # TODO: find a nicer way to put this in
        else:
            self.crisis = 0
        return self.crisis

    def _sell_stock(self, index, action, exercised_actions):
        """
        Perform sell actions.
        @param index            : asset index, points to the concerned asset
        @param action           : policy action, for one asset, as given by the agent. list, shape = (1)
        @param exercised_actions: used for saving the actually exercised actions for each asset.
                                  list, shape = (1 x number of assets)
        @return                 : exercised_actions, filled wth actual selling actions for each asset
        """

        def _sell_normal(index, action, exercised_actions):
            """
            If we hold assets, we can sell them based on our policy actions, but under short-selling constraints.
            Hence, we cannot sell more assets than we own.
            If we don't hold any assets, we pass, since we cannot short-sell.
            """
            # if self.state[index + self.assets_dim + 1] > 0:
            # if self.state[index + 1] > 0:
            if self.state["n_asset_holdings"][index] > 0:
                # todo: document; I have changed everywhere into round() because else actions (n.stocks to buy would be floats!)
                # todo: changed how state is constructed etc.
                # based on short-selling constraints, get actually exercisable action based on policy action and current asset holdings
                # exercised_action = min(round(abs(action)), self.state[index + self.assets_dim + 1])
                # exercised_action = min(round(abs(action)), self.state[index + 1])
                exercised_action = min(round(abs(action)), self.state["n_asset_holdings"][index])
                exercised_actions[index] = -exercised_action
                # update cash balance; cash new = cash old + price * n_assets sold*(1 - transaction cost)
                # self.state[0] += self.state[index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                # self.state[0] += self.state[self.assets_dim + index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                self.state["cash"][0] += self.state[self.price_colname][index] * \
                                         exercised_action * (1 - self.transaction_fee_percent)
                # update asset holdings
                # self.state[self.assets_dim + index + 1] -= exercised_action
                # self.state[index + 1] -= exercised_action
                self.state["n_asset_holdings"][index] -= exercised_action
                # update transaction cost; cost + new cost(price * n_assets_sold * transaction fee)
                # self.cost += self.state[self.assets_dim + index + 1] * exercised_action * self.transaction_fee_percent
                self.cost += self.state[self.price_colname][index] * exercised_action * self.transaction_fee_percent
                # update sell trades counter
                self.sell_trades += 1
            else:  # if we hold no assets, we cannot sell any
                exercised_actions[index] = 0
                pass
            return exercised_actions

        def _sell_off(index, exercised_actions):
            """
            If we hold assets, we sell them all, independent of the actions given by the policy.
            If we hold no assets, we pass (nothing to sell).
            """
            # if self.state[index + self.assets_dim + 1] > 0:
            # if self.state[index + 1] > 0: # if holdings larger than 0; we can sell
            if self.state["n_asset_holdings"][index] > 0:  # if holdings larger than 0; we can sell
                # exercised_actions = _sell_off(index, exercised_actions) # todo: ?
                # exercised action = asset holdings (since we sell all we hold)
                # exercised_action = self.state[index + 1] # exercised action = asset holdings (since we sell all we hold)
                exercised_action = self.state["n_asset_holdings"][
                    index]  # exercised action = asset holdings (since we sell all we hold)
                exercised_actions[index] = -exercised_action
                # update cash; old cash + new cash(price*action*(1-cost)
                # self.state[0] += self.state[index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                # self.state[0] += self.state[self.assets_dim + index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                self.state["cash"][0] += self.state[self.price_colname][index] \
                                         * exercised_action * (1 - self.transaction_fee_percent)
                # update asset holdings to 0 (since selloff)
                # self.state[index + self.assets_dim + 1] = 0
                # self.state[index + 1] = 0
                self.state["n_asset_holdings"][index] = 0
                # update transaction cost: price * action * cost
                # self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
                # self.cost += self.state[self.assets_dim + index + 1] * exercised_action * self.transaction_fee_percent
                self.cost += self.state[self.price_colname][index] * exercised_action * self.transaction_fee_percent
                # update sell trades
                self.sell_trades += 1
            else:  # if we hold no assets, we cannot sell any
                exercised_actions[index] = 0
            return exercised_actions

        ### PERFORM SELLING
        if self.crisis_measure is None:  # if None, sell normally
            exercised_actions = _sell_normal(index, action, exercised_actions)
        elif self.crisis_measure == "turbulence":
            # if True, go on; if None or False, then not used.
            # if our turbulence is below threshold, sell normally;
            if self.crisis < self.crisis_threshold:
                exercised_actions = _sell_normal(index, action, exercised_actions)
            else:
                exercised_actions = _sell_off(index, exercised_actions)
                print(f"(env) Crisis threshold exceeded (sell all, {self.mode}.")
        else:
            print("ERROR (env, sell): crisis condition must be None or specified correctly (see doc).")
        return exercised_actions

    def _buy_stock(self, index, action, exercised_actions):
        """
        Perform buy actions.
        @param index            : asset index, points to the concerned asset
        @param action           : policy action, for one asset, as given by the agent. list, shape = (1)
        @param exercised_actions: used for saving the actually exercised actions for each asset.
                                  list, shape = (1 x number of assets)
        @return                 : exercised_actions, filled with actual buying actions for each asset
        """

        def _buy_normal(index, action, exercised_actions):
            """
            We buy assets based on our policy actions given, under budget constraints.
            We cannot borrow in this setting, hence we can only buy as many assets as we can afford.
            We cannot buy fraction of assets in this setting.
            """
            # max_n_assets_to_buy = cash balance / stock price, rounded to the floor (lowest integer)
            # max_n_assets_to_buy = self.state[0] // self.state[index + 1]
            # max_n_assets_to_buy = self.state[0] // self.state[self.assets_dim + index + 1]
            max_n_assets_to_buy = self.state["cash"][0] // self.state[self.price_colname][index]
            # using the policy actions and budget constraints, get the actually exercisable action
            exercised_action = min(max_n_assets_to_buy, round(action))
            exercised_actions[index] = exercised_action
            # update cash position: old cash - new cash(price * action * (1-cost))
            # self.state[0] -= self.state[index + 1] * exercised_action * (1 + self.transaction_fee_percent)
            # self.state[0] -= self.state[self.assets_dim + index + 1] * exercised_action * (1 + self.transaction_fee_percent)
            self.state["cash"][0] -= self.state[self.price_colname][index] * \
                                     exercised_action * (1 + self.transaction_fee_percent)
            # update asset holdings for the current asset: old holdings + action
            # self.state[index + self.assets_dim + 1] += exercised_action
            # self.state[index + 1] += exercised_action
            self.state["n_asset_holdings"][index] += exercised_action
            # update transaction cost counter: price * action * cost
            # self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
            # self.cost += self.state[self.assets_dim + index + 1] * exercised_action * self.transaction_fee_percent
            self.cost += self.state[self.price_colname][index] * exercised_action * self.transaction_fee_percent
            # update buy trades counter
            self.buy_trades += 1
            return exercised_actions

        ### PERFORM BUYING
        if self.crisis_measure is None:  # if None, buy normally
            exercised_actions = _buy_normal(index, action, exercised_actions)
        elif self.crisis_measure == "turbulence":  # if True, go on; if None or False, then not used.
            # print("Using {} as crisis measure (buy, val), crisis = {}, threshold = {}.".format(
            # self.crisis_measure, self.crisis, self.crisis_threshold))
            # if our turbulence is below threshold, buy normally depending on max_n_assets_to_buy;
            if self.crisis < self.crisis_threshold:
                exercised_actions = _buy_normal(index, action, exercised_actions)
            else:
                print(f"(env) Crisis threshold exceeded (cease buying, {self.mode}.")
                # if turbulence goes over threshold, just stop buying
                exercised_actions[index] = 0
        else:
            print("ERROR (env, buy): crisis condition must be None or specified correctly (see doc).")
        return exercised_actions

    def reset(self):
        """
        Reset the environment to its initializations.
        @return: initial state after reset
        """
        self.reset_counter += 1
        self.steps_counter = 0

        if self.initial:
            self.terminal_state = False
            # self._seed() # TODO: added, check if this works
            self.day = self.firstday
            self.data = self.df.loc[self.day, :]
            logging.warning(f"({self.mode}) reset_env, initial = True, day = {self.day}")
            self.datadate = list(self.data[dataprep_settings.DATE_COLUMN])[
                0]  # take first element of list of identical dates
            self.memories["datadates"] = self.datadate
            # initialize reward; update after each step (after new day sampled)
            self.reward = 0
            # initialize crisis measure; update after each step (after new day sampled)
            self.crisis = 0
            # initialize cost (transaction cost); update after each trade
            self.cost = 0
            # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
            self.buy_trades = 0
            self.sell_trades = 0
            self.n_asset_holdings = [0] * self.assets_dim
            # initialize current state
            # old
            # self.state = [self.initial_cash_balance] + \
            #             self.data[self.price_colname].values.tolist() + \
            #             [0] * self.assets_dim
            # for feature in self.features_list:
            #    self.state += self.data[feature].values.tolist()
            # new
            self.current_cash_balance = self.initial_cash_balance
            self.current_n_asset_holdings = [0] * self.assets_dim
            self.observed_asset_prices_list = self.data[self.price_colname].values.tolist()
            # self.state = [self.current_cash_balance] + \
            #              self.current_n_asset_holdings + \
            #              self.observed_asset_prices_list
            # for feature in self.features_list:
            #    self.state += self.data[feature].values.tolist()
            # newest:
            self.state = {"cash": [self.current_cash_balance],
                          "n_asset_holdings": self.current_n_asset_holdings}
            if self.iteration == 126 or self.iteration == 1:
                asset_names = self.data[dataprep_settings.ASSET_NAME_COLUMN].unique()
                self.state_header = ["cash"] + [s + "_n_holdings" for s in asset_names]
            for feature in self.features_list:  # now price included in features list
                # self.state += self.data[feature].values.tolist()
                # newest:
                self.state.update({feature: self.data[feature].values.tolist()})
                if self.iteration == 126 or self.iteration == 1:
                    suffix = "_" + feature
                    self.state_header += [s + suffix for s in asset_names]
            # print("INITIAL STATE: ", self.state)
            # create flattened state_flattened
            self.state_flattened = list(chain(*list(self.state.values())))

            ##### INITIALIZE MEMORIES / MEMORY TRACKERS

            # update after each step (after new day sampled)
            self.memories = {"datadates": [self.datadate],
                             "cash_value": [self.initial_cash_balance],
                             "portfolio_value": [self.initial_cash_balance],
                             # "total_assets_value_memory" : [self.initial_cash_balance],
                             "rewards": [self.reward],
                             "policy_actions": [],
                             "exercised_actions": [],
                             "transaction_cost": [],
                             "number_asset_holdings": [self.n_asset_holdings],
                             "sell_trades": [],
                             "buy_trades": [],
                             "state_memory": [self.state_flattened],
                             }
            if self.crisis_measure is not None:
                self.memories.update({"crisis_measures": [self.crisis],
                                      "crisis_thresholds": [self.crisis_threshold],
                                      "crisis_selloff_cease_trading": []})
            if self.iteration == 126 or self.iteration == 1:
                self.memories.update({"state_header": [self.state_header]})

        else:
            self.terminal_state = False
            self.day = self.firstday
            self.data = self.df.loc[self.day, :]
            logging.warning(f"({self.mode}) reset_env, initial = False, day = {self.day}")
            self.datadate = list(self.data[dataprep_settings.DATE_COLUMN])[
                0]  # take first element of list of identical dates
            self.memories["datadates"] = self.datadate
            # initialize reward; update after each step (after new day sampled)
            self.reward = 0
            # initialize crisis measure; update after each step (after new day sampled)
            self.crisis = 0
            # initialize cost (transaction cost); update after each trade
            self.cost = 0
            # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
            self.buy_trades = 0
            self.sell_trades = 0

            # initialize state based on previous state
            # old
            # previous_total_asset = self.previous_state[0] + \
            #                       sum(np.array(self.previous_state[1:(self.assets_dim + 1)]) * np.array(
            #                           self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            # self.state = [self.previous_state[0]] + \
            #             self.data[self.price_colname].values.tolist() + \
            #             self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]
            # for feature in self.features_list:
            #    self.state += self.data[feature].values.tolist()
            # new
            # self.current_cash_balance = self.previous_state[0]
            self.current_cash_balance = self.previous_state["cash"][0]
            # self.current_n_asset_holdings = self.previous_state[1:(self.assets_dim + 1)]
            self.current_n_asset_holdings = self.previous_state["n_asset_holdings"]
            # previous_asset_prices = self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]
            previous_asset_prices = self.previous_state[self.price_colname]
            starting_portfolio_value = self.current_cash_balance + \
                                       sum(np.array(self.current_n_asset_holdings) * np.array(previous_asset_prices))
            # self.observed_asset_prices_list = self.data[self.price_colname].values.tolist() # integrated in features list
            # self.state = [self.current_cash_balance] + \
            #              self.current_n_asset_holdings + \
            #              self.observed_asset_prices_list
            # for feature in self.features_list:
            #    self.state += self.data[feature].values.tolist()
            # newest:
            self.state = {"cash": [self.current_cash_balance],
                          "n_asset_holdings": self.current_n_asset_holdings}
            for feature in self.features_list:  # now price included in features list
                self.state.update({feature: self.data[feature].values.tolist()})
            # create flattened state_flattened
            self.state_flattened = list(chain(*list(self.state.values())))

            self.n_asset_holdings = self.current_n_asset_holdings

            ##### INITIALIZE MEMORIES / MEMORY TRACKERS
            self.memories = {"datadates": [self.datadate],
                             "cash_value": [self.current_cash_balance],
                             # "portfolio_value": [previous_total_asset],
                             "portfolio_value": [starting_portfolio_value],
                             # "total_assets_value_memory" : [self.initial_cash_balance],
                             "rewards": [self.reward],
                             "policy_actions": [],
                             "exercised_actions": [],
                             "transaction_cost": [],
                             "number_asset_holdings": [self.n_asset_holdings],
                             "sell_trades": [],
                             "buy_trades": [],
                             "state_memory": [self.state_flattened],
                             }
            if self.crisis_measure is not None:
                self.memories.update({"crisis_measures": [self.crisis],
                                      "crisis_thresholds": [self.crisis_threshold],
                                      "crisis_selloff_cease_trading": []})
        return self.state_flattened

    def return_reset_counter(self, reset_counter):
        return self.reset_counter

    def render(self, mode='human', close=False):
        return [self.state_flattened, self.state, \
                self.reset_counter, self.final_state_counter, self.steps_counter]

    # this function creates a random seed
    # def _seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]
    def _seed(self, seed=None):
        return
