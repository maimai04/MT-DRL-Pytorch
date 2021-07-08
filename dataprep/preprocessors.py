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


def calculate_price_volume_OtherData() -> pd.DataFrame:
    """
    ## todo: define function to calculate price and trading volume for a
    @return:
    """
    pass


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
                trend-following momentum indicator, shows relationship between two moving averages of an asset's price
                macd is calculted by subtracting a 26-period (e.g. day) exponential moving average (EMA) from the
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

# todo: rm
def add_technical_indicator_with_otherFunc(df) -> pd.DataFrame:
    pass


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
    else:
        logging.info("add_other_features(): No features specified to add.")
    return df


def add_LSTM_features(df,
                     ann_model=None,
                     combine_with_df=True,
                     ) -> pd.DataFrame:
    """
    @param df:
    @param input_columns: list of columns fom the df to be used as lstm input (can be just price of each ticker,
                          or also technical indicators etc.
    @return:
    """

    # call trained ann model passed to function and get features
    # append features to original df
    if combine_with_df is False:
        # only combine the datadate and tic columns of the old dataframe with the new features
        pass

    else:
        # combine whole input df with all features created by the lstm
        pass

    return df


# Note: packed the below functions in one (don't really need two), was add_turbulence() and calcualte_turbuelnce() (merged them)

def split_data_by_date(df,
                       start,
                       end,
                       date_column="datadate",
                       asset_name_column="tic"
                       ) -> pd.DataFrame:  # OLD:renamed form data_split
    """
    split the dataset into training or testing using date.
    Used later mainly for train / validation / test split of the time series.

    INPUT: date subset df, sorted by date, then ticker,
    OUTPUT: date subset df, sorted by date, then ticker, and the index column is not [0,1,2,3,4,...] anymore
            but [0,0,0,0,...1,1,1,1,...,2,2,2,2,2...] etc. (datadate factorized, same index number for same datadate)
            This dataset is used then in the environment. So when we use day=0 in the env, we get day 0 for each ticker,
            hence multiple lines, not just one.
    @param df:
    @param start:
    @param end:
    @return:
    """
    # subsetting the dataframe based on date (start and end date)
    data = df[(df[date_column] >= start) & (df[date_column] < end)]
    # sorting the dataframe based on date, then based on ticker (company)
    # ignore_index=True ensures the new index will be re-labeled again in a sorted manner (1,2,3,...),
    # because by sorting the data set by datadate and tic, the initial index will not be sorted anymore
    #data = data.sort_values([date_column, asset_name_column])  # , ignore_index=True) # todo: not really needed, since index overwritten later anyways
    # data  = data[final_columns]
    # factorize the index of the dataframe, based on datadate
    # for the same datadate, the index will be the same, starting at 0 for the first date in the dataframe
    # .factorize returns a tuple of two lists; new index values, corresponding datadate values.
    # we only need the index values, hence the [0]
    #data.index = data[date_column].factorize()[0]

    return data


############################
##   COMBINED FUNCTIONS   ##
############################

# todo: was preprocess_data
def data_preprocessing_pipeline(  # BASE PARAMS FOR LOADING THE DATA SET - with load_dataset()
        raw_data_file=paths.RAW_DATA_FILE, # Note raw data file is ordered by ticker, then by date
        col_subset=data_settings.RAW_DF_COLS_SUBSET,
        date_subset="datadate",
        date_subset_startdate=settings.STARTDATE_TRAIN,

        # PASSING NAMES OF OPTIONAL FUNCTIONS TO BE USED
        calculate_price_volume_func="calculate_price_volume_WhartonData",
        add_technical_indicator_func="add_technical_indicator_with_StockStats",
        add_other_features_func=None,  # "add_other_features",
        add_LSTM_features_func=None,

        # PASSING PARAMETERS FOR EACH OPTIONAL FUNCTION
        # params for calculate_price_function()
        calculate_price_volume_func_params={"new_cols_subset": data_settings.NEW_COLS_SUBSET,
                                            "target_subset": None},
        # new_cols_subset=data_settings.NEW_COLS_SUBSET,
        # target_subset=None,

        # params for add_technical_indicator_func
        add_technical_indicator_func_params={"technical_indicators_list": ["macd", "rsi_30", "cci_30", "dx_30"]},
        # technical_indicators_list=["macd", "rsi_30", "cci_30", "dx_30"],

        # params for adding other features (e.g. volatility)
        add_other_features_func_params={"feature": ["returns_volatility", "log_return_daily"],
                                        "window_days": 7},

        # params for adding ANN-created features
        add_LSTM_features_func_params={},
        ) -> pd.DataFrame:
    """
    Data preprocessing pipeline: based on specifications, call preprocessing functions defined in preprocessors.py
    in a certain order and return a pre-processed dataframe.

    Note: TRAINING_DATA_FILE (raw df) is ordered based on ticker, not based on data;
    Example; raw df, ordered by ticker, then by date
        1   A
        2   A
        3   A
        1   B
        2   B
        3   B

    """
    # load the raw data set
    df = load_dataset(file_path=raw_data_file,
                      col_subset=col_subset,
                      date_subset=date_subset,
                      date_subset_startdate=date_subset_startdate)

    # get data after a specified date (originally was after 2009)
    # df = df[df["datadate"] >= startdate] # integrate din load_dataset
    # calculate adjusted price, open, high and low, and trading volume for each day
    logging.info("DATA PREPROCESSING PIPELINE:")
    logging.info("----------------------------")
    if calculate_price_volume_func == "calculate_price_volume_WhartonData":
        logging.info("data: Calculating price / volume on data from WhartonDB.")
        df = calculate_price_volume_WhartonData(df=df,
                                                new_cols_subset=calculate_price_volume_func_params["new_cols_subset"],
                                                target_subset=calculate_price_volume_func_params["target_subset"])
    elif calculate_price_volume_func == "calculate_price_volume_OtherData":
        logging.info("data: Calculating price / volume on alternative data (not WhartonDB).")
        df = calculate_price_volume_OtherData(df=df, )  # todo
    else:
        logging.info("DataPrep: No function specified for calculating price / volume from raw data.")

    # add technical indicators using the stockstats package
    if add_technical_indicator_func == "add_technical_indicator_with_StockStats":
        logging.info("data: technical indicators used (using stockstats package).")
        df = add_technical_indicator_with_StockStats(df=df,
                                                     technical_indicators_list=add_technical_indicator_func_params[
                                                         "technical_indicators_list"])
    elif add_technical_indicator_func == "add_technical_indicator_with_otherFunc":
        logging.info("DataPrep: technical indicators used (using other function).")
        df = add_technical_indicator_with_otherFunc(df=df,
                                                    )
    else:
        logging.info("DataPrep: No technical indicators used (because no function specified).")

    # add additional features such as volatility etc.
    if add_other_features_func == "add_other_features":
        logging.info("DataPrep: Added additional/other features (such as vola etc).")
        df = add_other_features(df=df,
                                features=add_other_features_func_params["feature"],
                                window_days=add_other_features_func_params["window_days"],
                                price_colum=data_settings.MAIN_PRICE_COLUMN,
                                asset_name_column=data_settings.ASSET_NAME_COLUMN
                                )
    else:
        logging.info("DataPrep: No additional features added (because no function specified).")

    # add additional features using an artificial neural network (trained model)
    if add_LSTM_features_func == "add_LSTM_features":
        logging.info("DataPrep: Adding additional features created with ANN.")
        df = add_LSTM_features(df=df,
                              ann_model=None,
                              combine_with_df=True)
    else:
        logging.info("DataPrep: No ANN-created features added.")
    df = df.sort_values(['datadate', 'tic']).reset_index(drop=True)
    # fill the missing values at the beginning
    df.fillna(method='bfill',
              inplace=True)  # TODO: this is for the tech indicators at the beginning, but could also drop!

    return df
