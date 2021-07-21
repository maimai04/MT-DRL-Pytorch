from stockstats import StockDataFrame
# own libraries:
from pipeline.setup_functions import *

############################
##   SINGLE FUNCTIONS   ##
############################


def calculate_price_volume_WhartonData(df,
                                       new_cols_subset=data_settings.NEW_COLS_SUBSET, # todo: rm
                                       target_subset=None,
                                       ):
    """
    Calculate the price and volume based on the Wharton dataset.  For other data sets, this function cannot be used
    directly (since different columns given), need to specify a different function.
    # todo: documentation and source for calculation (Wharton db)
    Depending on columns specified in new_cols_subset, the following columns are calculated like this: (see Wharton DB doc)
        adjcp (adjusted close price)    =  prccd / ajexdi
        open (day opening price)        =  prcod / ajexdi
        high (highest intraday price)   =  prchd / ajexdi
        low (lowest intraday price)     =  prcld / ajexdi
        volume (dily trading volume)    =  cashtrd

    @param df: raw pandas dataframe
    @return: (df) pandas dataframe with calculated

    INPUT: raw df as pd.DataFrame(), ordered by ticker, then date
    OUTPUT: df as pd.DataFrame(), with calculated price and volume, ordered by ticker, then date
    """

    data = df.copy()
    # data = data[raw_df_cols_subset] # to deprecate
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)  # todo-?
    # ajexdi is the "daily adjustment factor"; Adjusted close price = PRCCD/AJEXDI
    # https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiJxPKSnurvAhUmxIUKHUmHBV8QFjABegQIERAD&url=https%3A%2F%2Fwrds-www.wharton.upenn.edu%2Fdocuments%2F1402%2FCompustat_Global_in_WRDS_the_basics.pptx&usg=AOvVaw1_EFVBLEqobE1mltGZXrQd

    # this is the (mandatory) base columns subset
    base_subset = ['datadate', 'tic']

    # if we didn't specify a target_subset, the subset of columns to be returned with this function,
    # we automatically by default create a subset based on the base_subset columns and the new columns created below
    # we discard the "old" columns we used to create the new ones, however if we wanted, we could specify above
    # in the target_subset that we want to kep these columns as well by passing a lst of all the columns we want to keep
    if target_subset is None:
        target_subset = base_subset + new_cols_subset

    # calculate adjusted closing price # todo-?
    if "adjcp" in target_subset:
        data['adjcp'] = data['prccd'] / data['ajexdi']
    # calculate opening price # todo-?
    if "open" in target_subset:
        data['open'] = data['prcod'] / data['ajexdi']
    # calculate intraday high price # todo-?
    if "high" in target_subset:
        data['high'] = data['prchd'] / data['ajexdi']
    # calculate intraday low price # todo-?
    if "low" in target_subset:
        data['low'] = data['prcld'] / data['ajexdi']
    # calculate daily trading volume # todo-?
    if "volume" in target_subset:
        data['volume'] = data['cshtrd']

    #data = data[target_subset]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator_with_StockStats(df,
                                            technical_indicators_list=["macd", "rsi_30", "cci_30", "dx_30"]):
    # todo: was add_technical_indicator (specified as to "with stockstats", because not self-evident and matters for choice)
    """
    Calculate technical indicators, using the stockstats package:
    Stockstats documentation: https://pypi.org/project/stockstats/
    (comments and some changes done by me)

    @param df: pandas dataframe,
    @param technical_indicators_base:
    @return:

    Note: this function takes a few seconds to run (laptop)

    Indicators description:
    macd    :   moving average convergence divergence
                trend-following momentum indicator, shows relationship between two moving averages of an asset's price.
                macd is calculated by subtracting a 26-period (e.g. day) exponential moving average (EMA) from the
                12-period EMA
                Interpretation: # todo: ?
    rsi     :   relative strength index
                momentum indicator, measures magnitude of recent price changes to evaluate if the market is overbought
                or oversold
    cci     :   commodity channel index
                market indicator used to track market movements that may indicate buying or selling
                compares current price to average price ove a specific time period
                # todo: more
                # calculation:
                (Typical Price - Simple Moving Average) / (0.015 x Mean Deviation)
    dx      :   directional movement index, also dmi, adx
                measures strength and direction of price movement
                intended to reduce false signals
                adx: measures the strength of the up or down trend

    INPUT: df as pd.DataFrame(), with calculated price and volume, sorted by ticker, then date
    OUTPUT: df as pd.DataFrame(), with additional technical indicators for each asset, ordered by ticker, then date
    """
    df = df.sort_values(by=[data_settings.ASSET_NAME_COLUMN, data_settings.DATE_COLUMN])

    if technical_indicators_list is not None:
        # Converting the pandas df to a StockDataFrame using the retype function
        stock_df = StockDataFrame.retype(df.copy())
        # Note: the doc. says "this package takes for granted that your data is sorted by timestamp and contains
        # certain columns. Please align your column name: (open, close, high, low, volume, amount)".
        # => therefore, we need to rename our adjcp column into "close"
        stock_df['close'] = stock_df['adjcp']
        unique_ticker = stock_df.tic.unique()

        # temp = stock_df[stock_df.tic == unique_ticker[0]]['macd']
        for t_ind in technical_indicators_list:
            temp_df = pd.DataFrame()

            for i in range(len(unique_ticker)):
                temp_ind = stock_df[stock_df.tic == unique_ticker[i]][t_ind]
                temp_ind = pd.DataFrame(temp_ind)
                temp_df = temp_df.append(temp_ind, ignore_index=True)

            df[t_ind] = temp_df
    else:
        logging.info("add_technical_indicator_with_StockStats(): No features specified to add.")

    return df

def add_other_features(df,
                       features=["returns_volatility"],
                       window_days_vola=7,
                       min_periods_vola=1,
                       price_colum=data_settings.MAIN_PRICE_COLUMN,
                       asset_name_column=data_settings.ASSET_NAME_COLUMN
                       ) -> pd.DataFrame:
    """
    @param feature: price_volatility, returns_volatility

    INPUT: df as pd.DataFrame(), with additional technical indicators for each asset, sorted by ticker, then date
    OUTPUT: df as pd.DataFrame(), with additional features for each asset, ordered by ticker, then date
    """
    df = df.sort_values(by=[asset_name_column, data_settings.DATE_COLUMN])
    unique_ticker = df[asset_name_column].unique()

    if "price_volatility" in features:
        temp_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_ind = df[df.tic == unique_ticker[i]][price_colum].rolling(window_days_vola,
                                                                           min_periods=min_periods_vola
                                                                           ).std()
            temp_ind = pd.DataFrame(temp_ind)
            temp_df = temp_df.append(temp_ind, ignore_index=True)
        df["price_volatility"] = temp_df
    if "returns_volatility" in features:
        temp_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_ind = df[df.tic == unique_ticker[i]][price_colum].pct_change().rolling(window_days_vola,
                                                                                        min_periods=min_periods_vola
                                                                                        ).std()
            temp_ind = pd.DataFrame(temp_ind)
            temp_df = temp_df.append(temp_ind, ignore_index=True)
        df["returns_volatility"] = temp_df
    if "return_daily" in features:
        temp_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_ind = df[df.tic == unique_ticker[i]][price_colum].pct_change()
            temp_ind = pd.DataFrame(temp_ind)
            temp_df = temp_df.append(temp_ind, ignore_index=True)
        df["return_daily"] = temp_df
    if "log_return_daily" in features:
        temp_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_ind = np.log(df[df.tic == unique_ticker[i]][price_colum]) - \
                       np.log(df[df.tic == unique_ticker[i]][price_colum].shift(1))
            temp_ind = pd.DataFrame(temp_ind)
            temp_df = temp_df.append(temp_ind, ignore_index=True)
        df["log_return_daily"] = temp_df
    if "obv" in features:
        # calculate OBV for volume
        # see also: https://stackoverflow.com/questions/52671594/calculating-stockss-on-balance-volume-obv-in-python
        # https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/on-balance-volume-indicator-obv/
        # the obv is calculated like this:
        # take the sign of the differenced closing price
        # multiply this with the volume
        # the nan are filled with 0
        # then we take the cumulative sum over time
        # so basically, the obv = previous_day_obv + /- volume (if no change in price, it is 0* volume, hence only
        # previous-day obv)
        temp_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_ind = (np.sign(df[df.tic == unique_ticker[i]]["adjcp"].diff()) * df[df.tic == unique_ticker[i]]["volume"]).fillna(0).cumsum()
            temp_ind = pd.DataFrame(temp_ind)
            temp_df = temp_df.append(temp_ind, ignore_index=True)
        df["obv"] = temp_df
    else:
        logging.info("add_other_features(): No features specified to add.")
    return df




