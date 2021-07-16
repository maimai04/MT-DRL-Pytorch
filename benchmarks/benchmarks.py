import numpy as np
import pandas as pd
from config.config import *

def equal_weights_pf(data: pd.DataFrame,
                     rebalancing: int = 0,
                     cash_constraint: bool = True,
                     startdate: int = settings.STARTDATE_TRAIN,
                     enddate: int = settings.ENDDATE_TRADE,
                     cash: int = env_params.INITIAL_CASH_BALANCE,
                     transaction_fee = env_params.TRANSACTION_FEE_PERCENT,
                     no_fractions_tradable: bool = True
                     ):
    """
    Calculate a equally weighted portfolio (= equally cash-weighted)
    @param df: pandas df, time series of asset prices. Must be of the same assets and over
        same time line as used for individual PF (which we want to test against the benchmark)
        format: preprocessed data; sorted by datadate, ticker
    @param rebalancing: in which intervals to rebalance the portfolio to target weights.
        1: daily rebalancing, 2: rebalancing every second day, ..., len(df)+1: no rebalancing
    @return:
    """
    asset_names_col = dataprep_settings.ASSET_NAME_COLUMN
    date_col = dataprep_settings.DATE_COLUMN
    price_col = dataprep_settings.MAIN_PRICE_COLUMN

    if startdate is None:
        startdate = data[date_col].iloc[0]
    if enddate is None:
        enddate = data[date_col].iloc[-1]

    # get data by date and prepare for simpler time indexing
    data = data[[date_col, asset_names_col, price_col]]
    data = data[(data[date_col] >= startdate) & (data[date_col] <= enddate)]
    data = data.sort_values([date_col, asset_names_col])
    data.index = data[date_col].factorize()[0]

    # get weights vector
    asset_names = data[asset_names_col].unique()
    n_assets = len(asset_names)

    ### unique entry per day (index), each after rebalancing has been done
    data["initial_cash"] = cash
    data["initial_assets_value_total"] = 0
    data["initial_pf_value"] = data["initial_cash"] + data["initial_assets_value_total"]

    data["resulting_cash"] = 0
    data["resulting_assets_value_total"] = 0
    data["resulting_pf_value"] = 0
    data["transaction_cost_total"] = 0 # sum of transaction costs of each day

    ### unique entry for each asset each day
    # target weights of day t, calc. based on prices and cash of day t
    data["target_weights"] = 1/n_assets
    data["target_weights_delta"] = 0
    # weights day t, before daily rebalancing, with allocation from day t but new asset prices of day t+1
    data["weights_before_reb"] = 0
    # weights day t after daily rebalancing as close as possible to target weights
    data["weights_after_reb"] = 0
    # how much money was targeted to be invested in each asset at day t (using weights corrected for trans. cost)
    data["target_investment_delta"] = 0
    # how much money was actually invested in each asset at day t
    data["resulting_investment_delta"] = 0

    # how much was invested at the beginning of the day in a certain asset before rebalancing
    data["initial_investment"] = 0
    # how much much did we want to have invested in total in a certain asset after rebalancing
    data["target_investment"] = 0
    # how much much did we get to have invested in total in a certain asset after rebalancing
    data["resulting_investment"] = 0


    # how many units of each assets were meant to be traded (int) to reach the target weights
    # (using weights corrected for trans. cost)
    data["target_n_assets_traded"] = 0
    # how many units of each asset were actually traded (due to cash constraints)
    data["resulting_n_assets_traded"] = 0
    # how much of each asset to hold each day was our target (using weights corrected for trans. cost)
    data["target_n_asset_holdings"] = 0
    # how much of each asset do we hold each day
    data["resulting_n_asset_holdings"] = 0
    data["initial_n_asset_holdings"] = 0




    # at day one, buy assets according to weights_vector.
    # track cash value, transaction_fee and action taken
    for day in list(data.index.unique()):
        ### BEFORE TRADING
        # GET INITIAL VALUES / UPDATE INITIAL VALUES
        if day >= 1:
            data.loc[data.index == day, "initial_cash"] = data.loc[data.index == day-1, "resulting_cash"].values
            data.loc[data.index == day, "initial_n_asset_holdings"] = data.loc[data.index == day-1,
                                                                               "resulting_n_asset_holdings"].values
            data.loc[data.index == day, "initial_investment"] = \
                (data.loc[data.index == day, price_col] * data.loc[data.index == day, "initial_n_asset_holdings"]).values
            data.loc[data.index == day, "initial_pf_value"] = data.loc[data.index == day, "initial_cash"].values[0] + \
                                                              sum(data.loc[data.index == day, "initial_investment"])
        # single values
        initial_pf_value = data.loc[data.index == day, "initial_pf_value"].values[0]
        if day >= 1:
            data.loc[data.index == day, "weights_before_reb"] = (data.loc[data.index == day, "initial_investment"] / \
                                                                 initial_pf_value).values

        # lists (one entry for each asset)
        prices = data.loc[data.index == day, price_col].values.tolist()
        weights_before_reb = data.loc[data.index == day, "weights_before_reb"].values.tolist()
        initial_asset_holdings = data.loc[data.index == day, "initial_n_asset_holdings"].values.tolist()
        initial_investment = data.loc[data.index == day, "initial_investment"].values.tolist()
        # GET TARGETS
        # lists (one entry for each asset)
        target_weights = data.loc[data.index == day, "target_weights"].values.tolist()
        target_weights_delta = [(x1 - x2) for (x1, x2) in zip(target_weights, weights_before_reb)]
        target_investment = [x * initial_pf_value for x in target_weights]
        target_investment_delta = [x * initial_pf_value for x in target_weights_delta]
        target_n_assets_traded = [x1/x2 for (x1, x2) in zip(target_investment_delta, prices)]
        target_n_asset_holdings = (data.loc[data.index == day, "initial_n_asset_holdings"] +
                                   target_n_assets_traded).values.tolist()
        ### CONDUCT TRADE
        # min(target_assets_traded, target_assets_traded_after_tc)
        # sell: min(-50/price, -50*(1-0.01)/price) = -50/price n.stocks sold independent of transaction fee
        # buy: min(50/price, 50*(1-0.01)/price) = 50*(1-0.01) = 49.5 (transaction costs for buying add to buying costs)
        # target_cash_investment_after_tc = [x * (1-transaction_fee) for x in target_cash_investment]
        # => only "buy assets" affected by the fee
        # lists (one entry for each asset)
        resulting_n_assets_traded = [min(x1//x2, x1*(1-transaction_fee)//x2) for (x1, x2)
                                     in zip(target_investment_delta, prices)]
        # for sell, we can not sell more assets than we own (short-selling constraints)
        # for buy, we can sell independent from our number currently held
        # hence the max(N_assets_to_trade, -N_assets_owned);
        # if we want to sell 5 stocks but own only 2; max(-5, -2) = -2, if we own 20; max(-5,-20) = -5.
        # if we want to buy 5 stocks, will be independent of stock holdings; max(N>0, -N_owned) always N.
        # => only sell assets affected by this constraint (no short-selling)
        resulting_n_assets_traded = [max(x1, -x2) for (x1, x2) in zip(resulting_n_assets_traded, initial_asset_holdings)]
        resulting_n_asset_holdings = (data.loc[data.index == day, "initial_n_asset_holdings"] +
                                      resulting_n_assets_traded).values.tolist()
        resulting_investment_delta = [x1 * x2 for (x1, x2) in zip(resulting_n_assets_traded, prices)]
        resulting_investment = [x1 + x2 for (x1, x2) in zip(initial_investment, resulting_investment_delta)]
        transaction_cost = [x * transaction_fee for x in resulting_investment_delta]
        weights_after_reb = [x / initial_pf_value for x in resulting_investment]

        # values for current day, aggregated over whole portfolio
        transaction_cost_total = sum(transaction_cost)
        resulting_assets_value_total = sum(resulting_investment)
        resulting_cash = initial_pf_value - resulting_assets_value_total - transaction_cost
        resulting_pf_value = resulting_assets_value_total + resulting_cash

        # save results to the respective df subset
        # lists
        data.loc[data.index == day, "target_weights"] = target_weights
        data.loc[data.index == day, "target_weights_delta"] = target_weights_delta
        data.loc[data.index == day, "target_investment"] = target_investment
        data.loc[data.index == day, "target_investment_delta"] = target_investment_delta
        data.loc[data.index == day, "target_n_assets_traded"] = target_n_assets_traded
        data.loc[data.index == day, "target_n_asset_holdings"] = target_n_asset_holdings
        data.loc[data.index == day, "resulting_n_assets_traded"] = resulting_n_assets_traded
        data.loc[data.index == day, "resulting_n_asset_holdings"] = resulting_n_asset_holdings
        data.loc[data.index == day, "resulting_investment_delta"] = resulting_investment_delta
        data.loc[data.index == day, "resulting_investment"] = resulting_investment
        data.loc[data.index == day, "transaction_cost"] = transaction_cost
        data.loc[data.index == day, "weights_after_reb"] = weights_after_reb
        # single values
        data.loc[data.index == day, "resulting_assets_value_total"] = resulting_assets_value_total
        data.loc[data.index == day, "resulting_cash"] = resulting_cash
        data.loc[data.index == day, "resulting_pf_value"] = resulting_pf_value
        data.loc[data.index == day, "transaction_cost_total"] = transaction_cost_total

    # finally, calculate daily pf return and daily pf log return
    #data["daily_pf_return"] = data["daily_pf_return"]
    return data

def volume_weights_pf():
    pass

def index_pf():
    pass

def vola_weights_pf():
    pass

def random_weights_pf():
    pass

def mcap_weights_pf():
    pass
