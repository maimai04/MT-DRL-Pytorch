import numpy as np
import pandas as pd
from gym.utils import seeding
import gym # from openai
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config.config import paths, crisis_settings, settings, env_params, dataprep_settings, ppo_params



class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym

    see stable-baselines documentation on custom environments:
    https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html

    Note: no turbulence index here


    # shares normalization factor
    # 100 shares per trade
    hmax_normalize = 100
    # initial amount of money we have in our account
    initial_account_balance=1000000
    # total number of stocks in our portfolio
    assets_dim = 30
    # transaction fee: 1/1000 reasonable percentage
    transaction_fee_percent = 0.001
    reward_scaling = 1e-4
    """
    # BCAP: stock trading env. that follows the gyme interface (class Env, see core.py on GitHub:
    # https://github.com/openai/gym/blob/master/gym/core.py
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 assets_dim: int,
                 shape_observation_space: int,
                 features_list=dataprep_settings.FEATURES_LIST,
                 day=0,

                 # used for printout only
                 model_name="",
                 iteration="",

                 hmax_normalize=env_params.HMAX_NORMALIZE,
                 initial_account_balance=env_params.INITIAL_ACCOUNT_BALANCE,
                 transaction_fee_percent=env_params.TRANSACTION_FEE_PERCENT,
                 reward_scaling=env_params.REWARD_SCALING,

                 results_dir=None,
                 seed=settings.SEED_ENV
                 ):
        """

        @param df:
        @param features_list:
        @param day:
        @param iteration:
        @param model_name:
        @param hmax_normalize:
        @param initial_account_balance:
        @param assets_dim:
        @param transaction_fee_percent:
        @param reward_scaling:
        @param shape_observation_space:
        @param results_dir:
        @param seed:
        """
        ##### INPUT VARIABLES TO THE CLASS
        self.df = df
        self.features_list = features_list
        self.day = day
        self.iteration = iteration
        self.model_name = model_name

        self.hmax_normalize = hmax_normalize
        self.initial_account_balance = initial_account_balance
        self.assets_dim = assets_dim
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling
        self.shape_observation_space = shape_observation_space

        self.results_dir = results_dir

        ##### CREATING ADDITIONAL CLASS-INTERNAL VARIABLES
        self.data = self.df.loc[self.day, :]
        # action_space normalization and shape is assets_dim
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.assets_dim, ))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.shape_observation_space,))

        ##### INITIALIZATIONS

        # set terminal state to false at initialization
        self.terminal = False

        #self.state = [initial_account_balance] + \
        #              self.data.adjcp.values.tolist() + \
        #              [0]*assets_dim + \
        #              self.data.macd.values.tolist() + \
        #              self.data.rsi.values.tolist() + \
        #              self.data.cci.values.tolist() + \
        #              self.data.adx.values.tolist()
        self.state = [self.initial_account_balance] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * self.assets_dim
        for feature in self.features_list:
            self.state += self.data[feature].values.tolist()
        # initialize reward
        self.reward = 0
        # initialize cost (transaction cost)
        self.cost = 0
        # initialize trades counter
        self.trades = 0 # todo: deprecation ifo seebelow
        # initialize sell and buy trades counters
        self.buy_trades = 0
        self.sell_trades = 0

        ##### INITIALIZE MEMORIES / MEMORY TRACKERS
        # initialize total asset value memory to save whole asset value (incl. cash) trajectory
        self.total_asset_memory = [self.initial_account_balance]
        # initialize asset value memory w/o cash
        self.total_asset_memory = [self.initial_account_balance]
        # asset holdings memory # TODO: check if initialization at 0 or at variable
        self.asset_holdings_memory = []
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

        #self.reset()
        #self.seed(seed)
        
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        # print(actions)

        ##### IF WE ARE IN THE TERMINAL STATE #####
        if self.terminal:
            """
            If we are in the terminal state, we simply calculate the final values for
                cash balance
                asset values (excl. cash)
                total asset values (incl. cash)
                total df value (asset memory, total asset values over whole trajectory)
                actions trajectory
                
                
            """
            #plt.plot(self.total_asset_memory, 'r')
            #plt.savefig('{}/account_value_train.png'.format(self.results_dir))
            #plt.close()

            # final cash balance
            cash_balance_final = self.state[0]
            # final asset value (excl. cash) = sum of each stock price * n. stocks held
            asset_values_final = sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                     np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            # final portfolio value (incl. cash)
            end_total_asset = cash_balance_final + asset_values_final
            
            #print("end_total_asset:{}".format(end_total_asset))
            df_total_value = pd.DataFrame(self.total_asset_memory)
            df_total_value.to_csv('{}/01train_account_value_train_{}_{}.csv'.format(self.results_dir,
                                                                            self.model_name,
                                                                            self.iteration))
            #print("total_reward:{}".format(cash_balance_final+sum(np.array(self.state[1:(assets_dim+1)])*np.array(self.state[(assets_dim+1):61]))- initial_account_balance ))
            #print("total_cost: ", self.cost)
            #print("total_trades: ", self.trades)
            #df_total_value.columns = ['account_value']
            #df_total_value['daily_return'] = df_total_value.pct_change(1)
            #sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ df_total_value['daily_return'].std()
            #print("Sharpe: ",sharpe)
            #print("=================================")
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('{}/rewards_train_{}_{}.csv'.format(self.results_dir,
                                                                  self.model_name,
                                                                  self.iteration))
            
            # print('total asset: {}'.format(cash_balance_final+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
            #with open('obs.pkl', 'wb') as f:  
            #    pickle.dump(self.state, f)
            
            return self.state, self.reward, self.terminal, {}

        else:
            # print(np.array(self.state[1:29]))

            actions = actions * self.hmax_normalize # tODO: WHAT DOES THis mean? this means just that we multiply actions * 100 (?)? shouldnt it be :100 to normalize?
            #actions = (actions.astype(int))
            
            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                    np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            #load next state
            # print("stock_shares:{}".format(self.state[29:]))
            #self.state =  [self.state[0]] + \
            #        self.data.adjcp.values.tolist() + \
            #        list(self.state[(assets_dim+1):(assets_dim*2+1)]) + \
            #        self.data.macd.values.tolist() + \
            #        self.data.rsi.values.tolist() + \
            #        self.data.cci.values.tolist() + \
            #        self.data.adx.values.tolist()
            self.state = [self.state[0]]+ \
                         self.data.adjcp.values.tolist() + \
                         list(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)])
            for feature in self.features_list:
                self.state += self.data[feature].values.tolist()
            
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(self.assets_dim + 1)]) *
                                  np.array(self.state[(self.assets_dim + 1):(self.assets_dim * 2 + 1)]))
            self.total_asset_memory.append(end_total_asset)
            #print("end_total_asset:{}".format(end_total_asset))
            
            self.reward = end_total_asset - begin_total_asset            
            # print("step_reward:{}".format(self.reward))
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index + self.assets_dim + 1] > 0:
            # calculate actual action; min. between action and possible action
            # (cannot sell more than we own on stocks, since no short-selling)
            n_stocks_sold = min(abs(action), self.state[index + self.assets_dim + 1])

            # new cash balance = old cash balance + stock price * n. stocks sold * (1-transaction fee)
            self.state[0] += self.state[index + 1] * n_stocks_sold * (1 - self.transaction_fee_percent)
            # new stock holdings = old holdings - n. stocks sold
            self.state[index + self.assets_dim + 1] -= n_stocks_sold
            # total transaction costs = stock price * n. stocks sold * transaction fee
            self.cost += self.state[index + 1] * n_stocks_sold * self.transaction_fee_percent

            # update trades counter
            self.trades += 1
            self.sell_trades += 1

        else:
            pass

    def _buy_stock(self, index, action):
        # //; "floor division, rounds the result down to the nearest whole number
        # available amount = old cash balance // stock price = number of stocks we can buy
        # TODO: rename "available amout, since the name is confusing
        available_amount = self.state[0] // self.state[index + 1]
        # print('available_amount:{}'.format(available_amount))
        n_stocks_bought = min(available_amount, action)

        # new cash balance = old cash balance - stock price * n. stocks bought * (1-transaction fee)
        self.state[0] -= self.state[index + 1] * n_stocks_bought * (1 + self.transaction_fee_percent)
        # new stock holdings = old holdings + n. stocks bought
        self.state[index + self.assets_dim + 1] += n_stocks_bought
        # total transaction costs = stock price * n. stocks bought * transaction fee
        self.cost += self.state[index + 1] * n_stocks_bought * self.transaction_fee_percent

        # update trades counter
        self.trades += 1
        self.buy_trades += 1

    def reset(self):
        self.total_asset_memory = [self.initial_account_balance]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        #self.state = [self.initial_account_balance] + \
        #              self.data.adjcp.values.tolist() + \
        #              [0]*self.assets_dim + \
        #              self.data.macd.values.tolist() + \
        #              self.data.rsi.values.tolist() + \
        #              self.data.cci.values.tolist() + \
        #              self.data.adx.values.tolist()
        self.state = [self.initial_account_balance] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * self.assets_dim
        for feature in self.features_list:
            self.state += self.data[feature].values.tolist()
        # iteration += 1 
        return self.state
    
    def render(self, mode='human'):
        return self.state

    # this function creates a random seed
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
