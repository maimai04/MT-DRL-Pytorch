import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import Env
from gym import spaces
import matplotlib

matplotlib.use('Agg')
from config.config import paths, crisis_settings, settings, dataprep_settings, env_params, ppo_params
import matplotlib.pyplot as plt


class StockEnvValidation(gym.Env):
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
        self.data = self.df.loc[self.day, :]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.assets_dim,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.shape_observation_space,))

        ##### INITIALIZATIONS

        # initialize reward; update after each step (after new day sampled)
        self.reward = 0
        # initialize crisis measure; update after each step (after new day sampled)
        self.crisis = 0
        # initialize cost (transaction cost); update after each trade
        self.cost = 0
        # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
        self.buy_trades = 0
        self.sell_trades = 0

        self.exercised_actions = [0] * assets_dim
        #self.exercised_actions = [0] * self.assets_dim


        ##### INITIALIZE MEMORIES / MEMORY TRACKERS

        # update after each step (after new day sampled)
        self.cash_memory = [self.initial_cash_balance]
        # initialize total asset value memory to save whole asset value (incl. cash) trajectory
        self.portfolio_value_memory = [self.initial_cash_balance] # todo: was total_asset_and_cash_value
        # initialize asset value memory w/o cash
        self.total_assets_value_memory = [self.initial_cash_balance]
        # number of assets held; memory # TODO: check if initialization at 0 or at variable
        #self.n_asset_holdings_memory = [self.initial_n_asset_holdings]
        # initialize memory for reward trajectory
        self.rewards_memory = []
        # cost memory (summarizes sell and buy trading transaction costs) # todo: is it realistic to assume same transaction cost for buying and selling?
        self.cost_memory = []
        # actually exercised actions memory
        # (actions the agent actually undertook given policy actions under constraints)
        self.exercised_actions_memory = []
        # policy actions memory (actions recommended by the policy)
        self.policy_actions_memory = []
        # number of sell / buy trades memory
        self.buy_trades_memory = []
        self.sell_trades_memory = []
        # crisis_measures at each day
        self.crisis_measures = [self.crisis]
        self.crisis_thresholds = [self.crisis_threshold]
        self.crisis_selloff_cease_trading = []

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
        # memorize all the total balance change
        # self.reset()
        # self._seed(seed)
        # print("validation env. seed: ", self._seed())
        # state memory
        self.state_memory = [self.state]

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
        if self.mode is not "train": print(f"mode: {self.mode}: step = terminal state: {self.terminal_state}.")

        # normalized actions need to be multiplied by 100 to get number of stocks to purchase
        actions = actions * self.hmax_normalize
        #print(f"actions in step function (env): {actions}")
        # save policy actions (actions)
        self.policy_actions_memory.append(actions)


        ##### IF WE ARE IN THE TERMINAL STATE #####
        if self.terminal_state:
            #print(f"mode: {self.mode}: step = terminal state: {self.terminal_state}, saving memories.")

            # portfolio value = cash + assets value of new state
            end_portfolio_value = self.state[0] + \
                              sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                  np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            self.portfolio_value_memory.append(end_portfolio_value)
            end_cash_value = self.state[0]
            self.cash_memory.append(end_cash_value)

            # SAVING MEMORIES TO CSV
            pd.DataFrame(self.portfolio_value_memory).to_csv('{}/portfolio_value/end_portfolio_value_nextDayOpening_{}_{}_i{}.csv'.
                                                             format(self.results_dir, self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.cash_memory).to_csv('{}/cash_value/cash_value_{}_{}_i{}.csv'.format(self.results_dir, self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.rewards_memory).to_csv('{}/rewards/rewards_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.policy_actions_memory).to_csv('{}/policy_actions/policy_actions_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.exercised_actions_memory).to_csv('{}/exercised_actions/exercised_actions_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.sell_trades_memory).to_csv('{}/sell_trades/sell_trades_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.buy_trades_memory).to_csv('{}/buy_trades/buy_trades_{}_{}_i{}.csv'. format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.cost_memory).to_csv('{}/transaction_cost/transaction_cost_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.crisis_thresholds).to_csv('{}/crisis_thresholds/crisis_threshold_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.crisis_measures).to_csv('{}/crisis_measures/crisis_measures_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.crisis_selloff_cease_trading).to_csv('{}/crisis_selloff_cease_trading_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))
            pd.DataFrame(self.state_memory).to_csv('{}/state_memory/state_memory_{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))

            #pd.DataFrame(self.number_asset_holdings_memory).to_csv('{}/number_asset_holdings/number_asset_holdings-{}_{}_i{}.csv'.format(self.results_dir,self.mode, self.model_name, self.iteration))


            # df_total_value.columns = ['account_value']
            # df_total_value['daily_return'] = df_total_value.pct_change(1)
            # sharpe = (4**0.5)*df_total_value['daily_return'].mean()/ df_total_value['daily_return'].std()
            # print("Sharpe: ",sharpe)
            # saving to csv

            return self.state, self.reward, self.terminal_state, {}

        ##### IF WE ARE NOT YET IN THE TERMINAL STATE #####
        else:
            # print(np.array(self.state[1:29]))
            # actions = (actions.astype(int))
            if self.crisis_measure is not None:
                if self.crisis >= self.crisis_threshold:
                    # if crisis condition is true, overwrite actions because we want to sell all assets off
                    # so we set our actions to -100 (max. possible number of asset selling)
                    actions = np.array([-self.hmax_normalize] * self.assets_dim)  # TODO: ?
                    print(f"crisis {self.crisis} >= crisis threshold {self.crisis_threshold}: SELL ALL AND CEASE TRADING.")
                    self.crisis_selloff_cease_trading.append(1)
                elif self.crisis < self.crisis_threshold:
                    self.crisis_selloff_cease_trading.append(0)

            begin_portfolio_value = self.state[0] + \
                                sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                    np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            # print("begin_portfolio_value:{}".format(begin_portfolio_value))
            argsort_actions = np.argsort(actions)
            # get assets to be sold (if action -)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            # get assets to be bought (if action +)
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            self.exercised_actions = [0] * self.assets_dim

            # TODO: Note: this iterates index by index (stock by stock) and sells / buys stocks in consecutive order.
            # todo: for buy: can it happen that one stock uses up all recources an dthe others then cannot be bought anymore, in the extreme case?
            # todo: is there a way to buy stocks based on their *fraction* in the portfolio, instead based on number of stocks? since one
            # todo: cannot buy /sell more than 100 stocks and the cash for buying is also limited
            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            ### UPDATE VALUES AFTER ACTION TAKEN

            # memories just after action
            self.exercised_actions_memory.append(self.exercised_actions)
            self.sell_trades_memory.append(self.sell_trades)
            self.buy_trades_memory.append(self.buy_trades)
            self.cost_memory.append(self.cost)

            ### NEW DAY SAMPLING

            # after taking the actions (sell, buy), we update to the next day and get the new data from the dataset
            self.day += 1
            self.data = self.df.loc[self.day, :]

            ### UPDATE VALUES FOR NEW DAY

            # new crisis measure
            if self.crisis_measure is not None:
                self.crisis = self._update_crisis_measure()
                self.crisis_measures.append(self.crisis)
                self.crisis_thresholds.append(self.crisis_threshold)
            # print(self.turbulence)

            # load next state
            # print("stock_shares:{}".format(self.state[29:]))
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
                         list(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)])
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()

            # final portfolio + cash value after the end of the day with new prices (move up # todo)
            end_portfolio_value = self.state[0] + \
                              sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                  np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))

            self.reward = end_portfolio_value - begin_portfolio_value
            self.portfolio_value_memory.append(end_portfolio_value)
            self.cash_memory.append(self.state[0])
            self.rewards_memory.append(self.reward)
            self.state_memory.append(self.state)

            self.reward = self.reward * self.reward_scaling
            # we want to get the transaction cost for each day, not accumulated over multiple days, same for trades
            self.cost = 0
            self.sell_trades = 0
            self.buy_trades = 0

        return self.state, self.reward, self.terminal_state, {}

    def _update_crisis_measure(self):
        # we update our crisis measure, if we have some
        # self.turbulence = self.data['turbulence'].values[0]
        if self.crisis_measure == "turbulence":
            self.crisis = self.data[self.crisis_measure].values[0]  # TODO: find a nicer way to put this in
        else:
            self.crisis = 0
        #print(f"crisis updated for mode {self.mode}: {self.crisis}.")
        return self.crisis

    def _sell_stock(self, index, action):
        if self.crisis_measure == "turbulence":
            # if True, go on; if None or False, then not used.
            #print(f"Using {self.crisis_measure} as crisis measure (sell, {self.mode}), crisis = { self.crisis}, threshold = {self.crisis_threshold}.")
            # if our turbulence is below threshold, sell normally;
            if self.crisis < self.crisis_threshold:
                if self.state[index + self.assets_dim + 1] > 0:
                    #print('(val) policy action for stock {}: {}, actually selling: {}'.format(index, action, min(abs(action), self.state[index + self.assets_dim + 1])))
                    # if we hold assets, we sell them according to our policy action
                    # under the constraint that we cannot sell more than we own (no short-selling)
                    # hence the function min(...,...) self.exercised_actions
                    exercised_action = min(round(abs(action)), self.state[index + self.assets_dim + 1])
                    # todo: document; I have changed everywhere into round() because else actions (n.stocks to buy would be floats!)
                    #print(f"exercised sell action, mode {self.mode}: ")
                    #print(exercised_action)
                    self.exercised_actions[index] = -exercised_action

                    # update cash balance; cash new = cash old + price * n_assets sold*(1 - transaction cost)
                    self.state[0] += self.state[index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                    # update state, cost and sell_trades counter
                    self.state[self.assets_dim + index + 1] -= exercised_action
                    self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
                    self.sell_trades += 1
                else:
                    # if we hold no assets, we cannot sell any
                    pass
                    # if our turbulence is above threshold, sell all and cease trading;
            else:
                #print(f"Crisis threshold of {self.crisis_threshold} exceeded as crisis of {self.crisis} (sell all, {self.mode}.")

                # if turbulence goes over threshold, just clear out all positions if we have some (>0)
                if self.state[index + self.assets_dim + 1] > 0:  # TODO: understand this condition
                    # update state (balance)
                    exercised_action = self.state[index + self.assets_dim + 1]
                    self.exercised_actions[index] = -exercised_action

                    self.state[0] += self.state[index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                    # update state (number of assets held), cost and sell_trades counter
                    self.state[index + self.assets_dim + 1] = 0
                    self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
                    self.sell_trades += 1
                else:
                    # if we hold no assets, we cannot sell any
                    pass
        elif self.crisis_measure is None:  # if None, sell normally
            if self.state[index + self.assets_dim + 1] > 0:
                # update balance
                #print('(val) recommended action for stock {}: {}, actually selling: {}'.format(index, action, min(abs(action), self.state[index + self.assets_dim + 1])))
                exercised_action = min(round(abs(action)), self.state[index + self.assets_dim + 1])
                self.exercised_actions[index] = exercised_action

                self.state[0] += self.state[index + 1] * exercised_action * (1 - self.transaction_fee_percent)
                self.state[index + self.assets_dim + 1] -= exercised_action
                self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
                self.sell_trades += 1
            else:
                # if we hold no assets, we cannot sell any
                pass
        else:
            print("ERROR (sell): crisis condition must be None or specified correctly (see doc).")

    def _buy_stock(self, index, action):

        # perform buy action based on the sign of the action
        if self.crisis_measure == "turbulence":  # if True, go on; if None or False, then not used.
            # print("Using {} as crisis measure (buy, val), crisis = {}, threshold = {}.".format(
            # self.crisis_measure, self.crisis, self.crisis_threshold))
            # if our turbulence is below threshold, buy normally depending on available_amount;
            if self.crisis < self.crisis_threshold:
                # available amount = cash balance / stock price, rounded to the floor (lowest integer)
                # # todo: rename available_amount in max_assets_to_buy
                available_amount = self.state[0] // self.state[index + 1]

                exercised_action = min(available_amount, round(action))
                self.exercised_actions[index] = exercised_action

                #print('available amount for buying (val):{}'.format(available_amount))
               # print('(val) recommended action for stock {}: {}, actually buying: {}'.format(index, action,min(available_amount,action)))
                # update balance
                self.state[0] -= self.state[index + 1] * exercised_action * (1 + self.transaction_fee_percent)
                self.state[index + self.assets_dim + 1] += exercised_action
                self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
                self.buy_trades += 1
            else:
                print(f"Crisis threshold exceeded (cease buying, {self.mode}.")
                # if turbulence goes over threshold, just stop buying
                pass
        elif self.crisis_measure is None:  # if None, buy normally
            available_amount = self.state[0] // self.state[index + 1]
            #print('available amount for buying:{}'.format(available_amount))
            #print('(val) recommended action for stock {}: {}, actually buying: {}'.format(index,action,min(available_amount,action)))
            # update balance
            exercised_action = min(available_amount, round(action))
            self.exercised_actions[index] = exercised_action

            self.state[0] -= self.state[index + 1] * exercised_action * (1 + self.transaction_fee_percent)
            self.state[index + self.assets_dim + 1] += exercised_action
            self.cost += self.state[index + 1] * exercised_action * self.transaction_fee_percent
            self.buy_trades += 1
        else:
            print("ERROR (buy, val): crisis condition must be None or specified correctly (see doc).")

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
            # initialize reward; update after each step (after new day sampled)
            self.reward = 0
            # initialize crisis measure; update after each step (after new day sampled)
            self.crisis = 0
            # initialize cost (transaction cost); update after each trade
            self.cost = 0
            # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
            self.buy_trades = 0
            self.sell_trades = 0

            self.state = [self.initial_cash_balance] + \
                         self.data[self.price_colname].values.tolist() + \
                         [0] * self.assets_dim
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()

            self.exercised_actions = [0] * self.assets_dim

            ##### INITIALIZE MEMORIES / MEMORY TRACKERS

            # update after each step (after new day sampled)

            self.cash_memory = [self.initial_cash_balance]
            # initialize total asset value memory to save whole asset value (incl. cash) trajectory
            self.portfolio_value_memory = [self.initial_cash_balance]  # todo: was total_asset_and_cash_value
            # initialize asset value memory w/o cash
            self.total_assets_value_memory = [self.initial_cash_balance]
            # number of assets held; memory # TODO: check if initialization at 0 or at variable
            # self.n_asset_holdings_memory = [self.initial_n_asset_holdings]
            # initialize memory for reward trajectory
            self.rewards_memory = []
            # cost memory (summarizes sell and buy trading transaction costs) # todo: is it realistic to assume same transaction cost for buying and selling?
            self.cost_memory = []
            # actually exercised actions memory
            # (actions the agent actually undertook given policy actions under constraints)
            self.exercised_actions_memory = [0] * self.assets_dim
            # policy actions memory (actions recommended by the policy)
            self.policy_actions_memory = []
            # number of sell / buy trades memory
            self.buy_trades_memory = []
            self.sell_trades_memory = []
            # crisis_measures at each day
            self.crisis_measures = [self.crisis]
            self.crisis_thresholds = [self.crisis_threshold]
            self.crisis_selloff_cease_trading = []
            self.state_memory = [self.state]

        else:
            self.terminal_state = False
            self.day = 0
            self.data = self.df.loc[self.day, :]
            # initialize reward; update after each step (after new day sampled)
            self.reward = 0
            # initialize crisis measure; update after each step (after new day sampled)
            self.crisis = 0
            # initialize cost (transaction cost); update after each trade
            self.cost = 0
            # initialize sell and buy trade counters (counts how often a sell or buy action is performed each day)
            self.buy_trades = 0
            self.sell_trades = 0

            previous_total_asset = self.previous_state[0] + \
                                   sum(np.array(self.previous_state[1:(self.assets_dim + 1)]) * np.array(
                                       self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            self.state = [self.previous_state[0]] + \
                         self.data[self.price_colname].values.tolist() + \
                         self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()

            ##### INITIALIZE MEMORIES / MEMORY TRACKERS

            self.cash_memory = [self.previous_state[0]]
            self.portfolio_value_memory = [previous_total_asset]
            self.total_assets_value_memory = [self.initial_cash_balance]
            # number of assets held; memory # TODO: check if initialization at 0 or at variable
            # self.n_asset_holdings_memory = [self.initial_n_asset_holdings]
            self.rewards_memory = []
            self.cost_memory = []
            self.exercised_actions_memory = []
            self.policy_actions_memory = []
            self.buy_trades_memory = []
            self.sell_trades_memory = []
            self.crisis_measures = [self.crisis]
            self.crisis_thresholds = [self.crisis_threshold]
            self.crisis_selloff_cease_trading = []

            self.state_memory = [self.state]

        return self.state

    def render(self, mode='human', close=False):
        return self.state

    # this function creates a random seed
    # def _seed(self, seed=None):
    #    self.np_random, seed = seeding.np_random(seed)
    #    return [seed]

    def _seed(self, seed=None):
        return
