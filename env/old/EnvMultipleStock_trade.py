import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from config.config import paths, crisis_settings, settings, env_params, dataprep_settings, ppo_params


matplotlib.use('Agg')


class StockEnvTrade(gym.Env):  # TODO: change in AssetEnvTrade, to make more general
    """BCAP:
    A stock trading environment for OpenAI gym

    Attributes
    ----------
    df                      :   time series with all features to include in the state space of the environment
    features_list           :   list of features for state space construction, using the df column names.
                                By default: ["adjcp", "macd", "rsi", "cci", "adx"]
                                ([adjusted closing price, macd30, rsi30, cci30, adx30])
    day                     :   Day in the environment, for which the state space is created.
                                By default 0 (points to the first date of the df).
                                their comments: turbulence index: 90-150 reasonable threshold. TURBULENCE_THRESHOLD = 140
    crisis_measure          :   By default: None, else "turbulence".
                                Insert name of the column in the data set you want to use as
                                a measure of crisis / turbulence / vola etc.
    crisis_threshold        :   By default None. If crisis_measure specified as "turbulence", crisis_threshold =
                                turbulence_threshold in the previous version.
    initial                 :   ?
    previous_state          :   ?
    model_name              :   string. Name of the current model, used for naming saved files.
    iteration               :   string. Number of iterations, used for naming saved files.
    results_dir             :   directory where results are saved.

    hmax_normalize          :   ? By default 100. Shares normalization factor. 100 shares per trade at max. # TODO: if unused, remove, else explain
    initial_cash_balance :   Cash balance at the beginning of the episode, when only asset held is cash.
                                By default 1000000. Used to buy assets in their resp. currency.
    assets_dim               :   Number of assets considered (=number of unique tickers in df).
                                30 # TODO: don't hardcode, infer from df based on unique number of tickers
    transaction_fee_percent :   Transaction fee for trading (sell, buy) proportional to volume.
                                1/1000 reasonable percentage in their paper.
                                By default 0.001 based on reasoning (see in ...) # TODO: if unused, remove, else explain scientific reasoning
    reward_scaling          :   ? By default 1e-4. # TODO: if unused, remove, else explain

    Additionally created variables for initialization of the environment:
    ----------------------------------------------------------------------

    data                :   load data from a pandas dataframe for one day, as specified by variable "day",
                            and with all tickers and columns. Dataframe must be pre-sorted based on [day, ticker].
    action_space        :   action_space normalization and shape is assets_dim
                            Shape = 181 =
                                [Current Balance]+[prices 1-30]+[owned shares 1-30] +
                                [macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
    observation_space   :

    terminal            :   initialized at False. True, if we are in the final state of the episode.
    state               :   Flattened array of all states for all companies for the day chosen.
                            Example of what it looks like (only for 3 companies for simplification):
                                [1000000 (initial account balance),
                                 adjcp_value company 1,
                                 adjcp_value company 2,
                                 adjcp_value company 3,
                                 macd_value company 1,
                                 macd_value company 2,
                                 macd_value company 3,
                                 macd_value company 1
    reward              :   initialized at 0.
    cost                :   initialized at 0. transaction cost (?)
    trades              :   initialized at 0. number of traded stocks?
    crisis              :   initialized at 0. (replacing "turbulence")
    asset_memory        :   initialized as list with one entry: cash AUM (initial_cash_balance).
                            At each time step, add new total account value (assets value + cash).
    rewards_memory      :   initialized as empty list.
                            At each time step, add new reward value.

    Methods
    -------
    __init__        : specifies variables global to the whole class. # TODO explanation
    _sell_stock()   : sell assets currently held.
    _buy_stock()    : buy new assets.
    step()          : take a step in the environment to receive the reward and next state.
    reset()         : reset the environment to initial state and parameters.
    render()        : render the environment. Returns the next state. # TODO;: check difference with step()
    _seed()         : If seed not specified, creates a seed and sets it with numpy.
                    # BCAP: for seeds, see doc: https://github.com/openai/gym/blob/master/gym/utils/seeding.py

    """

    metadata = {'render.modes': ['human']}  # TODO: ?

    def __init__(self,
                 df: pd.DataFrame,
                 assets_dim: int,
                 shape_observation_space: int,
                 features_list=dataprep_settings.FEATURES_LIST,
                 day=0,

                 # used for printout only
                 model_name="",
                 iteration="",

                 # turbulence_threshold=140,
                 crisis_measure=crisis_settings.CRISIS_MEASURE,
                 crisis_threshold=None,

                 hmax_normalize=env_params.HMAX_NORMALIZE,
                 initial_cash_balance=env_params.INITIAL_CASH_BALANCE,
                 transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                 reward_scaling=env_params.REWARD_SCALING,

                 # params not in validation and training env
                 initial=True,  # BCAP: if we are in the initial state or not; TODO: do we need this var. and why?
                 previous_state=[],  # TODO: specify as list()?

                 # directories, saving paths
                 price_colname=dataprep_settings.MAIN_PRICE_COLUMN,
                 results_dir=None,  # BCAP: directory where results are saved # # TODO: specify as :str
                 seed=settings.SEED_ENV
                 ):

        ##### INPUT VARIABLES TO THE CLASS
        self.df = df
        self.features_list = features_list
        self.day = day
        self.crisis_measure = crisis_measure
        self.crisis_threshold = crisis_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.iteration = iteration
        self.hmax_normalize = hmax_normalize
        self.initial_cash_balance = initial_cash_balance
        self.assets_dim = assets_dim  # TODO: don't hardcode, infer from df based on unque number of tickers
        self.shape_observation_space = shape_observation_space
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling
        self.price_colname = price_colname
        self.results_dir = results_dir

        ##### CREATING ADDITIONAL CLASS-INTERNAL VARIABLES
        self.data = self.df.loc[self.day, :]
        self.action_space = spaces.Box(low=-1, high=1, shape=(assets_dim,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.shape_observation_space,))

        ##### INITIALIZATIONS

        # set terminal state to false at initialization
        self.terminal = False

        # self.state = [initial_account_balance] + \
        #             self.data.adjcp.values.tolist() + \
        #             [0] * assets_dim + \
        #             self.data.macd.values.tolist() + \
        #             self.data.rsi.values.tolist() + \
        #             self.data.cci.values.tolist() + \
        #             self.data.adx.values.tolist()
        self.state = [self.initial_cash_balance] + \
                     self.data[self.price_colname].values.tolist() + \
                     [0] * self.assets_dim
        for feature in self.features_list:
            self.state += self.data[feature].values.tolist()
        # initialize reward
        self.reward = 0
        # initialize cost (transaction cost)
        self.cost = 0
        # initialize trades counter
        self.trades = 0 # todo: deprecation ifo see below
        # initialize sell and buy trades counters
        self.buy_trades = 0
        self.sell_trades = 0
        # initialize crisis measure
        self.crisis = 0

        ##### INITIALIZE MEMORIES / MEMORY TRACKERS
        # initialize total asset value memory to save whole asset value (incl. cash) trajectory
        self.total_asset_and_cash_memory = [self.initial_cash_balance]
        # initialize asset value memory w/o cash
        self.total_asset_memory = [self.initial_cash_balance]
        # asset holdings memory # TODO: check if initialization at 0 or at variable
        self.number_asset_holdings_memory = []
        # initialize memory for reward trajectory
        self.rewards_memory = []
        # cost memory
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

        # self.reset()
        #self._seed(seed)
        #print("trading env. seed: ", self._seed())

    def step(self, actions):
        """BCAP:

        Function defines what happens when agent takes one step in the environment.
        At the end of each step, if the terminal state is reached, the account value is saved as csv to results_dir.

        Attributes:
        -----------
        actions:

        """
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # BCAP: note; we start with day 0, hence the -1
        # returns a True or False for terminal
        # print(actions)

        ##### IF WE ARE IN THE TERMINAL STATE #####
        if self.terminal:  # if we have reached the last step of the episode;
            plt.plot(self.asset_memory, 'r') # TODO: rm these lines wp
            plt.savefig('{}/account_value_trade_{}_{}.png'.format(self.results_dir,
                                                                  self.model_name,
                                                                  self.iteration))
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('{}/account_value_trade_{}_{}.csv'.format(self.results_dir,
                                                                            self.model_name,
                                                                            self.iteration))
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                  np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            print("total assets value start: {}".format(self.asset_memory[0]))
            print("total assets value end: {}".format(end_total_asset))
            print("total reward: {}".format(self.state[0] +
                                            sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                                np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)])) - self.asset_memory[0]))
            print("total cost: {}".format(self.cost))
            print("total trades: {}".format(self.trades))

            # df_total_value.columns = ['account_value'] # TODO: do we need this line?
            df_total_value['daily_return'] = df_total_value.pct_change(1)
            # TODO: check SR calculation; what is (4 ** 0.5) (srt(4))? and why is there no subtraction of the rf rate?
            # TODO: RED FLAG @ SR
            # internet: To get the annualized Sharpe ratio, you multiple the daily ratio by the square root of 252 (?)
            sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            print("Sharpe Ratio: {}".format(sharpe))

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('{}/rewards_trade_{}_{}.csv'.format(self.results_dir,
                                                                          self.model_name,
                                                                          self.iteration))

            return self.state, self.reward, self.terminal, {}

        ##### IF WE ARE NOT YET IN THE TERMINAL STATE #####
        else:
            # print(np.array(self.state[1:29]))
            actions = actions * self.hmax_normalize
            # actions = (actions.astype(int))
            if self.crisis_measure is not None:
                if self.crisis >= self.crisis_threshold:
                    # if crisis condition is true, overwrite actions because we have sold off all our assets
                    # and we have 0 assets, so we can
                    actions = np.array([-self.hmax_normalize] * self.assets_dim)  # TODO: ?
            ##
            # if self.turbulence_index_usage:
            # if self.turbulence >= self.turbulence_threshold:
            #   actions = np.array([-self.hmax_normalize] * self.assets_dim)  # BCAP: ?

            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(self.assets_dim + 1)]) * np.array(
                                    self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            # get assets to be sold (if action -)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            # get assets to be bought (if action +)
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                #print('Trading: recommended sell action: {}'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                #print('Trading: recommended buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            # update day by 1, when the next day starts#update day by 1, when the next day starts
            self.day += 1
            self.data = self.df.loc[self.day, :]

            # Note: the turbulence is already pre-calculated in pre-processing.
            # it is calculated as the turbulence of the whole index (all stocks) and hence
            # by asking for .values[0] we get the turbulence for the first tick in the list, here AAPL (Apple)
            # self.turbulence = self.data['turbulence'].values[0] # TODO: find a nicer way to put this in
            if self.crisis_measure is not None:
                self.crisis = self.data[self.crisis_measure].values[0]  # TODO: find a nicer way to put this in
            else:
                self.crisis = 0
            # print(self.turbulence)
            # load next state
            # print("stock_shares:{}".format(self.state[29:]))
            # TODO: automize so I don't have to add new states separately
            #self.state = [self.state[0]] + \
            #             self.data.[self.price_colname].values.tolist() + \
            #             list(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]) + \
            #             self.data.macd.values.tolist() + \
            #             self.data.rsi.values.tolist() + \
            #             self.data.cci.values.tolist() + \
            #             self.data.adx.values.tolist()
            self.state = [self.state[0]] + \
                         self.data[self.price_colname].values.tolist() +\
                         list(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)])
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()

            # Note:
            # self.state[0] = account balance
            # self.state[1:(self.assets_dim + 1) = adjusted closing prices (adjcp)
            # self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1) =
            # sum(np.array(self.state[1:(self.assets_dim + 1)]) = sum of all stock prices
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                  np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            self.asset_memory.append(end_total_asset)
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def _sell_stock(self, index, action):
        """BCAP:

        Function for selling an asset back to the environment.

        Attributes:
        -----------
        index   :   The index (row labels) of the Pandas DataFrame.
        action  :   # TODO find out what this looks like when program runs

        """
        # perform sell action based on the sign of the action

        # TODO: check if this works;
        # BCAP: created a more general function where we can pass "any" condition we want which leads to a
        # sell-out of all held assets into cash and halting of trading until another condition is fulfilled
        # and this condition is not fulfilled anymore

        if self.crisis_measure == "turbulence":  # if True, go on; if None or False, then not used.
            print("Using {} as crisis measure (sell, trade), crisis = {}, threshold = {}.".format(
                self.crisis_measure, self.crisis, self.crisis_threshold))
            # if our turbulence is below threshold, sell normally;
            if self.crisis < self.crisis_threshold:
                # changed: crisis instead of turbulence, crisis_threshold instead of turbulence_threshold
                if self.state[index + self.assets_dim + 1] > 0:
                    # calculate actual action
                    n_stocks_sold = min(abs(action), self.state[index + self.assets_dim + 1])

                    print('(trade) recommended action for stock {}: {}, actually selling: {}'.format(
                        index, action, n_stocks_sold))

                    # new cash balance = old cash balance + stock price * n. stocks sold * (1-transaction fee)
                    self.state[0] += self.state[index + 1] * n_stocks_sold * (1 - self.transaction_fee_percent)
                    # new stock holdings = old holdings - n. stocks sold
                    self.state[index + self.assets_dim + 1] -= n_stocks_sold
                    # total transaction costs = stock price * n. stocks sold * transaction fee
                    self.cost += self.state[index + 1] * n_stocks_sold * self.transaction_fee_percent

                    # update trades counter
                    self.trades += 1
                else:
                    # if we hold no assets, we cannot sell any
                    pass
            # if our turbulence is above threshold, sell all and cease trading;
            else:
                print("Crisis threshold of {} exceeded by crisis of {} (sell all, trade).".format(
                    self.crisis_threshold, self.crisis))
                # if turbulence goes over threshold, just clear out all positions if we have some (>0)
                if self.state[index + self.assets_dim + 1] > 0:
                    # update balance
                    self.state[0] += self.state[index + 1] * self.state[index + self.assets_dim + 1] * \
                                     (1 - self.transaction_fee_percent)
                    self.state[index + self.assets_dim + 1] = 0
                    self.cost += self.state[index + 1] * self.state[index + self.assets_dim + 1] * \
                                 self.transaction_fee_percent
                    self.trades += 1
                else:
                    # if we hold no assets, we cannot sell any
                    pass
        elif self.crisis_measure is None:  # if None, sell normally
            if self.state[index + self.assets_dim + 1] > 0:
                # update balance
                print('(trade) recommended action for stock {}: {}, actually selling: {}'.format(
                    index, action, min(abs(action), self.state[index + self.assets_dim + 1])))
                self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + self.assets_dim + 1]) * \
                                 (1 - self.transaction_fee_percent)
                self.state[index + self.assets_dim + 1] -= min(abs(action), self.state[index + self.assets_dim + 1])
                self.cost += self.state[index + 1] * min(abs(action), self.state[index + self.assets_dim + 1]) * \
                             self.transaction_fee_percent
                self.trades += 1
            else:
                # if we hold no assets, we cannot sell any
                pass
        else:
            print("ERROR (sell, trade): crisis condition must be None or specified correctly (see doc).")

    def _buy_stock(self, index, action):
        """
        BCAP:

        function parameters:
            index   :
            action  :

        specialty:
            If turbulence_index_usage = True, buying is ceased when turbulence is above or equal the turbulence index.

        """

        # perform buy action based on the sign of the action
        if self.crisis_measure == "turbulence":  # if True, go on; if None or False, then not used.
            print("Using {} as crisis measure (buy, trade), crisis = {}, threshold = {}.".format(
                self.crisis_measure, self.crisis, self.crisis_threshold))
            # if our turbulence is below threshold, buy normally depending on available_amount;
            if self.crisis < self.crisis_threshold:
                # changed: crisis instead of turbulence, crisis_threshold instead of turbulence_threshold
                available_amount = self.state[0] // self.state[index + 1] # REDFLAG: DOES NOT ACCOUNT FOR TRANSACTION COSTS WHICH WE NEED TO PAY AS WELL
                print('available amount for buying (trade): {}'.format(available_amount))
                print('(trade) recommended action for stock {}: {}, actually buying: {}'.format(index,
                                                                                                action,
                                                                                                min(available_amount, action)))
                # update balance
                self.state[0] -= self.state[index + 1] * min(available_amount, action) * (
                            1 + self.transaction_fee_percent)
                self.state[index + self.assets_dim + 1] += min(available_amount, action)
                self.cost += self.state[index + 1] * min(available_amount, action) * self.transaction_fee_percent
                self.trades += 1
            else:
                print("Crisis threshold of {} exceeded by crisis of {} (cease buying, trade).".format(
                    self.crisis_threshold, self.crisis))
                # if turbulence goes over threshold, just stop buying
                pass
        elif self.crisis_measure is None:  # if None, buy normally
            available_amount = self.state[0] // self.state[index + 1]
            print('available amount for buying (trade):{}'.format(available_amount))
            print('(trade) recommended action for stock {}: {}, actually buying: {}'.format(index,
                                                                                            action,
                                                                                            min(available_amount, action)))

            # update balance
            self.state[0] -= self.state[index + 1] * min(available_amount, action) * \
                             (1 + self.transaction_fee_percent)
            self.state[index + self.assets_dim + 1] += min(available_amount, action)
            self.cost += self.state[index + 1] * min(available_amount, action) * \
                         self.transaction_fee_percent
            self.trades += 1
        else:
            print("ERROR (buy): crisis condition must be None or specified correctly (see doc).")

    def reset(self):
        if self.initial:
            self.asset_memory = [self.initial_cash_balance]
            self.day = 0
            self.data = self.df.loc[self.day, :]

            # self.turbulence = 0
            self.crisis = 0

            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=self.iteration
            self.rewards_memory = []
            # initiate state
            # TODO: automize so I don't have to add new states separately
            # self.state = [initial_account_balance] + \
            #             self.data.adjcp.values.tolist() + \
            #             [0] * self.assets_dim + \
            #             self.data.macd.values.tolist() + \
            #             self.data.rsi.values.tolist() + \
            #             self.data.cci.values.tolist() + \
            #             self.data.adx.values.tolist()
            self.state = [self.initial_cash_balance] + \
                         self.data[self.price_colname].values.tolist() +\
                         [0] * self.assets_dim
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()
        else:
            previous_total_asset = self.previous_state[0] + \
                                   sum(np.array(self.previous_state[1:(self.assets_dim + 1)]) * np.array(
                                       self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            self.asset_memory = [previous_total_asset]
            # self.asset_memory = [self.previous_state[0]]
            self.day = 0
            self.data = self.df.loc[self.day, :]

            # self.turbulence = 0
            self.crisis = 0

            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=iteration
            self.rewards_memory = []
            # initiate state
            # self.previous_state[(assets_dim+1):(assets_dim*2+1)]
            # [0]*assets_dim + \

            #self.state = [self.previous_state[0]] + \
            #             self.data.adjcp.values.tolist() + \
            #             self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)] + \
            #             self.data.macd.values.tolist() + \
            #             self.data.rsi.values.tolist() + \
            #             self.data.cci.values.tolist() + \
            #             self.data.adx.values.tolist()
            self.state = [self.previous_state[0]] + \
                         self.data[self.price_colname].values.tolist() +\
                         self.previous_state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()

        return self.state

    def render(self, mode='human', close=False):
        return self.state

    # this function creates a random seed
    # BCAP: sources: https://stackoverflow.com/questions/58304804/random-seeding-in-open-ai-gym
    #def _seed(self, seed=None):
        # BCAP: "seed must be a non-negative integer or omitted (None))
     #   self.np_random, seed = seeding.np_random(seed)
        # this function then calls create:seed, which does the following:
        # """Create a strong random seed. Otherwise, Python 2 would seed using
        #     the system time, which might be non-robust especially in the
        #     presence of concurrency.
    #    return [seed]

    def _seed(self, seed=None):
        # BCAP: "seed must be a non-negative integer or omitted (None))
        #self.np_random, seed = seeding.np_random(seed)
        # this function then calls create:seed, which does the following:
        # """Create a strong random seed. Otherwise, Python 2 would seed using
        #     the system time, which might be non-robust especially in the
        #     presence of concurrency.
        return


