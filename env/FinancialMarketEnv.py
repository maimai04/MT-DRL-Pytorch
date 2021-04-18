import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import Env
from gym import spaces
import matplotlib
import logging

matplotlib.use('Agg')
from config.config import paths, crisis_settings, settings, dataprep_settings, env_params, ppo_params
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed


class FinancialMarketEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 assets_dim: int,
                 shape_observation_space: int,
                 features_list=dataprep_settings.FEATURES_LIST,
                 day=0,
                 mode="",  # "validation", "test" =trade

                 # used for printout only
                 model_name="",
                 iteration="",

                 # turbulence_threshold=140,
                 crisis_measure=crisis_settings.CRISIS_MEASURE,
                 crisis_threshold=0,

                 hmax_normalize=env_params.HMAX_NORMALIZE,
                 initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                 transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                 reward_scaling=env_params.REWARD_SCALING,

                 # params not in validation and training env
                 initial=True,  # BCAP: if we are in the initial state or not; TODO: do we need this var. and why?
                 previous_state=[],  # TODO: specify as list()?

                 price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                 results_dir=None,
                 seed=settings.SEED_ENV
                 ):

        ##### INPUT VARIABLES TO THE CLASS
        self.mode = mode
        self.df = df
        self.features_list = features_list
        self.day = day
        self.model_name = model_name
        self.iteration = iteration

        # only relevant for trade:
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

        ##### CREATING ADDITIONAL CLASS-INTERNAL VARIABLES

        # action_space normalization and shape is assets_dim
        self.data = self.df.loc[self.day, :] # includes all tickers, hence >1 line for >1 assets
        self.datadate = list(self.data[dataprep_settings.DATE_COLUMN])[0] # take first element of list of identical dates

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.assets_dim,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.shape_observation_space,))

        ##### INITIALIZATIONS
        # set terminal state to false at initialization
        self.terminal_state = False

        # initializing state for current day
        # self.state = [self.initial_account_balance] + \
        #             self.data.adjcp.values.tolist() + \
        #             [0]*assets_dim + \
        #             self.data.macd.values.tolist() + \
        #             self.data.rsi.values.tolist() + \
        #             self.data.cci.values.tolist() + \
        #             self.data.adx.values.tolist()
        self.state = [self.initial_cash_balance] + \
                     self.data[price_colname].values.tolist() + \
                     [0] * self.assets_dim  # todo: put this at the beginning, then the rest can be automized, but nee dto change calculations below tooo!
        for feature in self.features_list:
            self.state += self.data[feature].values.tolist()

        # self.reset()
        # self._seed(seed)

        # initialize reward; update after each step (after new day sampled)
        self.reward = 0
        # initialize crisis measure; update after each step (after new day sampled)
        self.crisis = 0
        # initialize cost (transaction cost); update after each trade
        self.cost = 0
        # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
        self.buy_trades = 0
        self.sell_trades = 0
        self.n_asset_holdings = 0

        ##### INITIALIZE MEMORIES / MEMORY TRACKERS
        # note: key names must be the same as for paths.SUBSUBDIR_NAMES, else saving to csv doesn't work (dirs won't match)
        self.memories = {"datadates" : [self.datadate],
                         "cash_value" : [self.initial_cash_balance],
                         "portfolio_value" : [self.initial_cash_balance], # starting value (beginning of the day), before taking a step
                         #"total_assets_value_memory" : [self.initial_cash_balance],
                         "rewards" : [self.reward], # reward in first entry is 0 because we can only calculate it after a day has passed (but could be done differnetly, doesn't matter)
                         "policy_actions": [],
                         "exercised_actions": [],
                         "transaction_cost" : [],
                         "number_asset_holdings": [self.n_asset_holdings],
                         "sell_trades": [],
                         "buy_trades" : [],
                         "state_memory" : [self.state],
                         }
        if self.crisis_measure is not None:
            self.memories.update({"crisis_measures": [self.crisis],
                                  "crisis_thresholds": [self.crisis_threshold],
                                  "crisis_selloff_cease_trading": []})

    def step(self, actions):
        """
        Check if we are in the terminal state.
            If yes, return the final state, reward...
            If no, take a step in environment and an action, then load next state and return reward, state...
        This function calls _sell_stock() and _buy_stock()

        @param actions:
        @return: self.state, self.reward, self.terminal_state, {}
        """
        self.terminal_state = self.day >= len(self.df.index.unique()) - 1

        # normalized actions need to be multiplied by 100 to get number of stocks to purchase
        actions = actions * self.hmax_normalize
        # save policy actions (actions), update policy memory which is independent of step taken
        self.memories["policy_actions"].append(actions)

        ##### IF WE ARE IN THE TERMINAL STATE #####
        if self.terminal_state:
            self.memories["exercised_actions"].append([0] * self.assets_dim) # because no actions exercised in this last date anymore
            self.memories["transaction_cost"].append(0) # since no transaction on this day, no transaction costcost
            self.memories["sell_trades"].append(0)
            self.memories["buy_trades"].append(0)

            # SAVING MEMORIES TO CSV
            pd.DataFrame({"datadate" : self.memories["datadates"]}
                         ).to_csv(os.path.join(self.results_dir, "datadates", f"datadates_{self.mode}_{self.model_name}_i{self.iteration}.csv"))
            for key in list(self.memories.keys())[1:]:
                #print(key)
                #print(f"key length: {len(self.memories[key])}, vs. datadate length: {len(self.memories['datadate_memory'])}.")
                pd.DataFrame({"datadate": self.memories["datadates"], key: self.memories[key]}).to_csv(
                    os.path.join(self.results_dir, key, f"{key}_{self.mode}_{self.model_name}_i{self.iteration}.csv"))

            # df_total_value.columns = ['account_value'] # todo: rm
            # df_total_value['daily_return'] = df_total_value.pct_change(1)
            # sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ df_total_value['daily_return'].std()
            # print("Sharpe: ",sharpe)
            # saving to csv

            return self.state, self.reward, self.terminal_state, {}

        ##### IF WE ARE NOT YET IN THE TERMINAL STATE #####
        else:
            if self.crisis_measure is not None:
                if self.crisis >= self.crisis_threshold:
                    # if crisis condition is true, overwrite actions because we want to sell all assets off
                    # so we set our actions to -100 (max. possible number of asset selling)
                    actions = np.array([-self.hmax_normalize] * self.assets_dim)
                    self.memories["crisis_selloff_cease_trading"].append(1)
                elif self.crisis < self.crisis_threshold:
                    self.memories["crisis_selloff_cease_trading"].append(0)

            begin_portfolio_value = self.state[0] + \
                                sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                    np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
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
            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                exercised_actions = self._sell_stock(index, actions[index], exercised_actions)
            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                exercised_actions = self._buy_stock(index, actions[index], exercised_actions)

            ### UPDATE VALUES AFTER ACTION TAKEN
            # counters to be changed after actions (apart from sell and buy traded,
            # which are change din the sell and buy functions directyl)
            self.n_asset_holdings = list(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)])
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
            self.datadate = list(self.data[dataprep_settings.DATE_COLUMN])[0]  # take first element of list of identical dates
            self.memories["datadates"].append(self.datadate)

            ### UPDATE VALUES FOR NEW DAY
            # new crisis measure
            if self.crisis_measure is not None:
                self.crisis = self._update_crisis_measure()
                self.memories["crisis_measures"].append(self.crisis)
                self.memories["crisis_thresholds"].append(self.crisis_threshold)

            # load next state
            # TODO: explain why changed
            # self.state =  [self.state[0]] + \ # TODO: here different than for trading, why?
            #       self.data.adjcp.values.tolist() + \
            #       list(self.state[(assets_dim+1):(assets_dim*2+1)]) + \ # TODO: here different than for trading, why?
            #       self.data.macd.values.tolist() + \
            #       self.data.rsi.values.tolist() + \
            #       self.data.cci.values.tolist() + \
            #       self.data.adx.values.tolist()
            # new state
            self.state = [self.state[0]] + \
                         self.data[self.price_colname].values.tolist() + \
                         list(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]) # n. asset hldings
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()
            # final portfolio + cash value after the end of the day with new prices (move up # todo)
            end_portfolio_value = self.state[0] + \
                              sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                  np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            self.reward = end_portfolio_value - begin_portfolio_value
            self.memories["rewards"].append(self.reward)
            self.memories["portfolio_value"].append(end_portfolio_value)
            self.memories["cash_value"].append(self.state[0])
            self.memories["state_memory"].append(self.state)
            # apply reward scaling
            self.reward = self.reward * self.reward_scaling
            # we want to get the transaction cost for each day, not accumulated over multiple days, same for trades
            self.cost = 0
            self.sell_trades = 0
            self.buy_trades = 0
        return self.state, self.reward, self.terminal_state, {}

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
            if self.state[index + self.assets_dim + 1] > 0:
                # based on short-selling constraints, get actually exercisable action based on policy action
                exercised_action = min(round(abs(action)), self.state[index + self.assets_dim + 1])
                # todo: document; I have changed everywhere into round() because else actions (n.stocks to buy would be floats!)
                exercised_actions[index] = -exercised_action
                # update cash balance; cash new = cash old + price * n_assets sold*(1 - transaction cost)
                self.state[0] += self.state[index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                # update asset holdings
                self.state[self.assets_dim + index + 1] -= exercised_action
                # update transaction cost
                self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
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
            if self.state[index + self.assets_dim + 1] > 0:
                exercised_actions = _sell_off(index, exercised_actions)
                exercised_action = self.state[index + self.assets_dim + 1]
                exercised_actions[index] = -exercised_action
                self.state[0] += self.state[index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                self.state[index + self.assets_dim + 1] = 0
                self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
                self.sell_trades += 1
            else: # if we hold no assets, we cannot sell any
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
        else:
            print("ERROR (sell): crisis condition must be None or specified correctly (see doc).")
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
            max_n_assets_to_buy = self.state[0] // self.state[index + 1]
            # using the policy actions and budget constraints, get the actually exercisable action
            exercised_action = min(max_n_assets_to_buy, round(action))
            exercised_actions[index] = exercised_action
            # update cash position
            self.state[0] -= self.state[index + 1] * exercised_action * (1 + self.transaction_fee_percent)
            # update asset holdings for the current asset
            self.state[index + self.assets_dim + 1] += exercised_action
            # update transaction cost counter
            self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
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
                print(f"Crisis threshold exceeded (cease buying, {self.mode}.")
                # if turbulence goes over threshold, just stop buying
                exercised_actions[index] = 0
        else:
            print("ERROR (buy, val): crisis condition must be None or specified correctly (see doc).")
        return exercised_actions

    def reset(self):
        """
        Reset the environment to its initializations.
        @return: initial state after reset
        """
        if self.initial:
            self.terminal_state = False
            # self._seed() # TODO: added, check if this works
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.datadate = list(self.data[dataprep_settings.DATE_COLUMN])[0]  # take first element of list of identical dates
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
            self.n_asset_holdings = 0
            # initialize current state
            self.state = [self.initial_cash_balance] + \
                         self.data[self.price_colname].values.tolist() + \
                         [0] * self.assets_dim
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()

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
                             "state_memory": [self.state],
                             }
            if self.crisis_measure is not None:
                self.memories.update({"crisis_measures": [self.crisis],
                                      "crisis_thresholds": [self.crisis_threshold],
                                      "crisis_selloff_cease_trading": []})
        else:
            self.terminal_state = False
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.datadate = list(self.data[dataprep_settings.DATE_COLUMN])[0]  # take first element of list of identical dates
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
            previous_total_asset = self.previous_state[0] + \
                                   sum(np.array(self.previous_state[1:(self.assets_dim + 1)]) * np.array(
                                       self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            self.state = [self.previous_state[0]] + \
                         self.data[self.price_colname].values.tolist() + \
                         self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()

            self.n_asset_holdings = self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]

            ##### INITIALIZE MEMORIES / MEMORY TRACKERS
            self.memories = {"datadates": [self.datadate],
                             "cash_value": [self.previous_state[0]],
                             "portfolio_value": [previous_total_asset],
                             # "total_assets_value_memory" : [self.initial_cash_balance],
                             "rewards": [self.reward],
                             "policy_actions": [],
                             "exercised_actions": [],
                             "transaction_cost": [],
                             "number_asset_holdings": [self.n_asset_holdings],
                             "sell_trades": [],
                             "buy_trades": [],
                             "state_memory": [self.state],
                             }
            if self.crisis_measure is not None:
                self.memories.update({"crisis_measures": [self.crisis],
                                      "crisis_thresholds": [self.crisis_threshold],
                                      "crisis_selloff_cease_trading": []})
        return self.state

    def render(self, mode='human', close=False):
        return self.state

    # this function creates a random seed
    # def _seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]
    def _seed(self, seed=None):
        return
