import numpy as np
import pandas as pd
from stockstats import StockDataFrame
from config.config import *
from config.config import settings, crisis_settings, paths, env_params, dataprep_settings
import logging

############################
##   SINGLE FUNCTIONS   ##
############################

def load_dataset(*,
                 file_path,
                 col_subset=None,  # dataprep_settings.RAW_DF_COLS_SUBSET,
                 date_subset=None,  # "datadate",
                 date_subset_startdate=None,  # 20090000
                 ) -> pd.DataFrame:
    """
    Load the .csv dataset from the provided file path.
    If a col_subset is specified, only the specified columns subset of the loaded dataset is returned.
    (This can be used if there are many columns and we only want to use 5 of them e.g.)

    @type file_path: object
    @param file_path as specified in config.py
    @return: (df) pandas dataframe

    # Note: the asterisk (*) enforces that all the following variables have to be specified as keyword argument, when being called
    # see also: https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/

    INPUT: df as csv, ordered by ticker, then date
    OUTPUT: df as pd.DataFrame(), ordered by ticker, then date
    """
    df = pd.read_csv(file_path, index_col=0)

    if col_subset is not None:
        df = df[col_subset]

    if date_subset and date_subset_startdate is not None:
        df = df[df[date_subset] >= date_subset_startdate]

    return df


# Note: was calculate_price()
def calculate_price_volume_WhartonData(df,
                                       new_cols_subset=dataprep_settings.NEW_COLS_SUBSET,
                                       target_subset=None
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
    @param raw_df_cols_subset: subset of raw dataset which we want to use for feature calculation etc.
    @param additional_subset: features to calculate and return these columns only as a subset of the full data set.
                              (adding the base features datadate and tic, which cannot be omitted).
                              Do not pass an empty list, unless, you will add other features later in add_features(),
                              (Make sure the agent has some data to learn apart from date and ticker)
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

    data = data[target_subset]
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

    INPUT: df as pd.DataFrame(), with calculated price and volume, ordered by ticker, then date
    OUTPUT: df as pd.DataFrame(), with additional technical indicators for each asset, ordered by ticker, then date
    """
    from stockstats import StockDataFrame

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


def add_technical_indicator_with_otherFunc(df) -> pd.DataFrame:
    pass


def add_other_features(df,
                       feature="returns_volatility",
                       window_days=7,
                       price_colum=dataprep_settings.MAIN_PRICE_COLUMN,
                       asset_name_column=dataprep_settings.ASSET_NAME_COLUMN
                       ) -> pd.DataFrame:
    """
    @param feature: price_volatility, returns_volatility

    INPUT: df as pd.DataFrame(), with additional technical indicators for each asset, ordered by ticker, then date
    OUTPUT: df as pd.DataFrame(), with additional features for each asset, ordered by ticker, then date
    """
    unique_ticker = df[asset_name_column].unique()

    if feature == "price_volatility":
        temp_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_ind = df[df.tic == unique_ticker[i]][price_colum].rolling(window_days, min_periods=1).std()
            temp_ind = pd.DataFrame(temp_ind)
            temp_df = temp_df.append(temp_ind, ignore_index=True)
        df[feature] = temp_df
    elif feature == "returns_volatility":
        temp_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            temp_ind = df[df.tic == unique_ticker[i]][price_colum].pct_change().rolling(window_days, min_periods=1).std()
            temp_ind = pd.DataFrame(temp_ind)
            temp_df = temp_df.append(temp_ind, ignore_index=True)
        df[feature] = temp_df
    else:
        logging.info("add_other_features(): No features specified to add.")

    return df


def add_ANN_features(df,
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
def add_crisis_measure(df,
                       crisis_measure=crisis_settings.CRISIS_MEASURE,
                       ) -> pd.DataFrame:
    """
    add turbulence index from a precalculated dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    @param df:
    @param crisis_measure:
    """

    """calculate turbulence index based on dow 30"""
    if crisis_measure == "turbulence":
        df_price_pivot = df.pivot(index='datadate', columns='tic', values='adjcp')
        unique_date = df.datadate.unique()
        # start after a year
        start = 252  # todo
        turbulence_index = [0] * start
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index]]
            cov_temp = hist_price.cov()
            current_temp = (current_price - np.mean(hist_price, axis=0))
            temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        crisis_index = pd.DataFrame({'datadate': df_price_pivot.index, 'turbulence': turbulence_index})
        df = df.merge(crisis_index, on='datadate')

    elif crisis_measure is None:
        logging.info("add_crisis_measure(): settings: no crisis measure used.")
    else:
        logging.info("add_crisis_measure(): no valid crisis measure defined.")

    return df


def get_data_params(final_df: pd.DataFrame,  # todo: create support_functions and move there
                    base_cols=dataprep_settings.BASE_DF_COLS,
                    feature_cols=dataprep_settings.FEATURES_LIST,
                    asset_name_column="tic",
                    date_column="datadate",
                    startdate_validation=settings.STARTDATE_VALIDATION,
                    enddate_validation=settings.ENDDATE_VALIDATION,
                    ) -> list:
    """
    Get some parameters we need, based on the final pre-processed dataset:
        number of individual assets (n_individual_assets)
        number of features used (n_features)
        unique trade dates within the wished validation (or other) subset (unique_trade_dates)
    @param final_df:
    @param base_cols:
    @param asset_name_column:
    @param date_column:
    @param startdate_validation:
    @param enddate_validation:
    @return:
    """
    df = final_df.copy()
    n_individual_assets = len(df[asset_name_column].unique())
    n_features = len(feature_cols)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_dates needs to start from 2015/10/01 for validation purpose
    unique_trade_dates = df[(df[date_column] > startdate_validation) &
                            (df[date_column] <= enddate_validation)][date_column].unique()
    # TODO: IS THIS TRUE ? (DATE = ENDDATE VALIDATION? OR ENDDATE TRADING?)

    return [n_individual_assets, n_features, unique_trade_dates]

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
        col_subset=dataprep_settings.RAW_DF_COLS_SUBSET,
        date_subset="datadate",
        date_subset_startdate=settings.STARTDATE_TRAIN,

        # PASSING NAMES OF OPTIONAL FUNCTIONS TO BE USED
        calculate_price_volume_func="calculate_price_volume_WhartonData",
        add_technical_indicator_func="add_technical_indicator_with_StockStats",
        add_other_features_func=None,  # "add_other_features",
        add_ANN_features_func=None,
        add_crisis_measure_func="add_crisis_measure",

        # PASSING PARAMETERS FOR EACH OPTIONAL FUNCTION
        # params for calculate_price_function()
        calculate_price_volume_func_params={"new_cols_subset": dataprep_settings.NEW_COLS_SUBSET,
                                            "target_subset": None},
        # new_cols_subset=dataprep_settings.NEW_COLS_SUBSET,
        # target_subset=None,

        # params for add_technical_indicator_func
        add_technical_indicator_func_params={"technical_indicators_list": ["macd", "rsi_30", "cci_30", "dx_30"]},
        # technical_indicators_list=["macd", "rsi_30", "cci_30", "dx_30"],

        # params for adding other features (e.g. volatility)
        add_other_features_func_params={"feature": "returns_volatility",
                                        "window_days": 7},

        # params for adding ANN-created features
        add_ANN_features_func_params={},

        # params for add_crisis_measure_func
        add_crisis_measure_func_params={"crisis_measure": crisis_settings.CRISIS_MEASURE},
        # crisis_measure=crisis_settings.CRISIS_MEASURE,
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
        logging.info("DataPrep: Calculating price / volume on data from WhartonDB.")
        df = calculate_price_volume_WhartonData(df=df,
                                                new_cols_subset=calculate_price_volume_func_params["new_cols_subset"],
                                                target_subset=calculate_price_volume_func_params["target_subset"])
    elif calculate_price_volume_func == "calculate_price_volume_OtherData":
        logging.info("DataPrep: Calculating price / volume on alternative data (not WhartonDB).")
        df = calculate_price_volume_OtherData(df=df, )  # todo
    else:
        logging.info("DataPrep: No function specified for calculating price / volume from raw data.")

    # add technical indicators using the stockstats package
    if add_technical_indicator_func == "add_technical_indicator_with_StockStats":
        logging.info("DataPrep: technical indicators used (using stockstats package).")
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
                                feature=add_other_features_func_params["feature"],
                                window_days=add_other_features_func_params["window_days"],
                                price_colum=dataprep_settings.MAIN_PRICE_COLUMN,
                                asset_name_column=dataprep_settings.ASSET_NAME_COLUMN
                                )
    else:
        logging.info("DataPrep: No additional features added (because no function specified).")

    # add additional features using an artificial neural network (trained model)
    if add_ANN_features_func == "add_ANN_features":
        logging.info("DataPrep: Adding additional features created with ANN.")
        df = add_ANN_features(df=df,
                              ann_model=None,
                              combine_with_df=True
                              )
    else:
        logging.info("DataPrep: No ANN-created features added.")

    if add_crisis_measure_func == "add_crisis_measure":
        logging.info("DataPrep: crisis measure function called.")
        df = add_crisis_measure(df=df,
                                crisis_measure=add_crisis_measure_func_params["crisis_measure"])
    else:
        logging.info("DataPrep: no crisis measure calculated.")
    df = df.sort_values(['datadate', 'tic']).reset_index(drop=True)
    # fill the missing values at the beginning
    df.fillna(method='bfill',
              inplace=True)  # TODO: this is for the tech indicators at the beginning, but could also drop!

    return df


def get_crisis_threshold(df,
                         mode="insample",
                         crisis_measure=crisis_settings.CRISIS_MEASURE,
                         date_colname=dataprep_settings.DATE_COLUMN,
                         crisis_measure_colname=crisis_settings.CRISIS_MEASURE,
                         cutoff_Xpercentile=crisis_settings.CUTOFF_XPERCENTILE,
                         insample_data_turbulence_threshold=None,
                         insample_data_subset=None,
                         startdate=settings.STARTDATE_TRAIN,
                         enddate=settings.STARTDATE_VALIDATION,
                         ) -> list:
    """

    @param mode:
    @param insample_data_turbulence_threshold:
    @param df:
    @param crisis_measure:
    @param startdate:
    @param enddate:
    @param date_colname:
    @param crisis_measure_colname:
    @param cutoff_Xpercentile:
    @return:
    """
    data = df.copy()
    logging.info("GET CRISIS THRESHOLD:")
    logging.info("---------------------")
    if mode == "insample":
        logging.info(f"-get_crisis_threshold (insample), mode: {mode}, crisis measure: {crisis_measure}.")
        if crisis_measure == "turbulence":
            """
            CALCULATE INSAMPLE TURBULENCE THRESHOLD BASED ON THE STARTING TRAINING DATA
            this is the initial training data, used only for training, before getting extended window by window later
            FOR COMPARISON WITH CURRENT (HISTORICAL) TURBULENCE LATER
            """
            # turbulence_threshold = 140 # todo
            insample_data_subset = data[(data[date_colname] >= startdate) &
                                        (data[date_colname] < enddate)].drop_duplicates(subset=[date_colname])
            insample_data_turbulence_threshold = np.quantile(insample_data_subset[crisis_measure_colname].values,
                                                             cutoff_Xpercentile)
            # returns a float; the insample turbulence threshold at 90th percentile
            # with default data: df[(df.datadate<20151000) & (df.datadate>=20090000)], gives 96.08
            logging.info("-insample turbulence threshold: {}".format(insample_data_turbulence_threshold))
            to_return = [insample_data_turbulence_threshold, insample_data_subset]

        elif crisis_measure is None:
            logging.info("-no crisis measure used (a).")
            to_return = [0, None]
        else:
            logging.info("-ValueError: specified crisis measure does not exist (a).")
            to_return = [None, None]

    elif mode == "newdata":
        logging.info(f"-get_crisis_threshold mode {mode}, crisis measure {crisis_measure}.")
        if crisis_measure == "turbulence":
            # logging.info("crisis measure used: ", crisis_settings.CRISIS_MEASURE )
            current_turbulence = data[(data.datadate >= startdate) & (data.datadate <= enddate)]
            current_turbulence = current_turbulence.drop_duplicates(subset=[date_colname])
            current_turbulence_mean = np.mean(current_turbulence[crisis_measure_colname].values)
            # old:
            # historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
            # historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])
            # historical_turbulence_mean = np.mean(current_turbulence.turbulence.values)
            if current_turbulence_mean > insample_data_turbulence_threshold:
                # if the mean of the current data is greater than the 90% quantile of insample turbulence data,
                # then we assume that the current market is volatile,
                # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
                # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
                turbulence_threshold = insample_data_turbulence_threshold
                logging.info("-current turbulence index mean used: {}".format(turbulence_threshold))
                to_return = [turbulence_threshold, None]
            else:
                # if the mean of the historical data is less than the 90% quantile of insample turbulence data
                # then we tune up the turbulence_threshold, meaning we lower the risk
                turbulence_threshold = np.quantile(insample_data_subset[crisis_measure_colname].values, 1)
                logging.info("-90th percentile of insample turbulence used: {} ", turbulence_threshold)
                to_return = [turbulence_threshold, None]
                # todo: why???

        elif crisis_measure is None:
            logging.info("-no crisis measure used (b).")
            turbulence_threshold = 0
            to_return = [turbulence_threshold, None]
        else:
            logging.info("-ValueError: specified crisis measure does not exist (b).")
            turbulence_threshold = 0
            to_return = [None, None]
    else:
        logging.info("-unknown mode specification in get_crisis_threshold().\nMust be 'insample' or 'new'.")
        turbulence_threshold = 0
        to_return = [None, None]

    return to_return
