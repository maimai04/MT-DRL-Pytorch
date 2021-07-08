import time
# RL models from stable-baselines
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import gym
import math
import glob
import ffn
import pandas as pd
import numpy as np

# own libraries
from config.config import *
from environment.FinancialMarketEnv import FinancialMarketEnv
from model.CustomOnPolicyBuffer import OnPolicyBuffer
from model.CustomPPOAlgorithm import PPO_algorithm
from model.CustomActorCriticNets import BrainActorCritic, \
    init_weights_feature_extractor_net, init_weights_actor_net, init_weights_critic_net

########################################################################
# DEFINE FUNCTIONS PERFORMANCE EVALUATION                              #
########################################################################

def calculate_performance_measures(run_path: str, # path where the target results are saved for the whole run
                                   # whether the results are calculated for one seed or across all seeds (whole run)
                                   seed: int,
                                   level="seed", #"run"
                                   mode: str = "test",
                                   logger=None,
                                   ) -> None:
    logger.info("Starting Calculation of Performance Measures.")
    # if we aggregate performances for one seed
    if level == "seed":
        # we get the path to the performance files for this current seed
        #seed_path = os.path.join(run_path, "randomSeed"+str(seed))
        seed_path = run_path #os.path.join(run_path, "agentSeed" + str(seed)) # todo: deprec.
        backtest_path = os.path.join(seed_path, "backtest")
        # Note: here, run_path already containst the seed folder

        # get results for one seed
        results_dict, backtest_dict, _, _, _ = get_results_dict_for_one_seed(seed_path=seed_path,
                                                                             backtest_path=backtest_path,
                                                                             mode="test")

        calculate_and_save_performance_metrics(results_dict=results_dict,
                                               backtest_dict=backtest_dict,
                                               save_path=seed_path,
                                               seed=seed,
                                               mode="test")
    # if we aggregate performances for all seeds
    if level == "run":
        li = []
        # for every seed in the list of seeds
        for seed in settings.SEEDS_LIST:
            seed_path = os.path.join(run_path, "randomSeed"+str(seed))
            #seed_path = os.path.join(run_path, "agentSeed" + str(seed)) # todo: deprec.
            # get the performance metrics summary df for the current seed
            # Note: here, run_path does not contain the seed oflder but is one level higher,
            # therefore we need to create the seed path for each seed first
            df = pd.read_csv(glob.glob(os.path.join(seed_path,
                                                    f"*{mode}_performance_metrics_seed{seed}.csv"))[0], index_col=0)
            df.set_index("performance_metric", inplace=True)
            li.append(df)
        df = pd.concat(li, axis=1)
        # calculate average and median over all seeds
        df["mean"] = df.mean(axis=1)
        df["median"] = df.median(axis=1)
        df["min"] = df.min(axis=1)
        df["max"] = df.max(axis=1)
        df["std"] = df.std(axis=1)
        # save to csv under the path of the hwole run
        df.to_csv(os.path.join(run_path, f"{mode}_performance_metrics_allSeeds.csv"))
    logger.info("Finished Calculation of Performance Measures.")
    return None

def get_results_dict_for_one_seed(seed_path: str,
                                  backtest_path: str,
                                  mode="test",
                                  ) -> Tuple[dict, dict, pd.DataFrame, dict, dict]:
    # performance paths for FOLDERS
    pfvalue_path = os.path.join(seed_path, "portfolio_value")
    reward_path = os.path.join(seed_path, "rewards")
    all_weights_path = os.path.join(seed_path, "all_weights_cashAtEnd")
    equity_weights_path = os.path.join(seed_path, "asset_equity_weights")
    policy_actions_path = os.path.join(seed_path, "policy_actions")
    exer_actions_path = os.path.join(seed_path, "exercised_actions")
    state_mem_path = os.path.join(seed_path, "state_memory")
    # training performance
    training_performance_path = os.path.join(seed_path, "training_performance")
    # for backtesting
    bt_pfvalue_path = os.path.join(backtest_path, "portfolio_value")
    bt_reward_path = os.path.join(backtest_path, "rewards")
    bt_all_weights_path = os.path.join(backtest_path, "all_weights_cashAtEnd")
    bt_equity_weights_path = os.path.join(backtest_path, "asset_equity_weights")
    bt_policy_actions_path = os.path.join(backtest_path, "policy_actions")
    bt_exer_actions_path = os.path.join(backtest_path, "exercised_actions")
    bt_state_mem_path = os.path.join(backtest_path, "state_memory")
    # create dictionarys with results paths of the folders aggregated
    results_dict = {"pfvalue": glob.glob(os.path.join(pfvalue_path, f"*{mode}*.csv")),
                   "reward": glob.glob(os.path.join(reward_path, f"*{mode}*.csv")),
                   "all_weights": glob.glob(os.path.join(all_weights_path, f"*{mode}*.csv")),
                   "equity_weights": glob.glob(os.path.join(equity_weights_path, f"*{mode}*.csv")),
                   "policy_actions": glob.glob(os.path.join(policy_actions_path, f"*{mode}*.csv")),
                   "exer_actions": glob.glob(os.path.join(exer_actions_path, f"*{mode}*.csv")),
                   "state_mem": glob.glob(os.path.join(state_mem_path, f"*{mode}*.csv")),
                   }
    backtest_dict = {"pfvalue": glob.glob(os.path.join(bt_pfvalue_path, f"*{mode}*.csv")),
                   "reward": glob.glob(os.path.join(bt_reward_path, f"*{mode}*.csv")),
                   "all_weights": glob.glob(os.path.join(bt_all_weights_path, f"*{mode}*.csv")),
                   "equity_weights": glob.glob(os.path.join(bt_equity_weights_path, f"*{mode}*.csv")),
                   "policy_actions": glob.glob(os.path.join(bt_policy_actions_path, f"*{mode}*.csv")),
                   "exer_actions": glob.glob(os.path.join(bt_exer_actions_path, f"*{mode}*.csv")),
                   "state_mem": glob.glob(os.path.join(bt_state_mem_path, f"*{mode}*.csv")),
                   }
    state_header_df = pd.read_csv(os.path.join(state_mem_path, "state_header.csv"), index_col=0)
    # get the data from the paths saved in the dictionary
    ### FOR RESULTS DICTIONARY
    results_dicty = results_dict.copy()
    for key in results_dicty:
        # for every key in results_dicty (e.g. "pfvalue", "reward",...) we get the list of filepaths
        # (there are multiple filepaths since we have multiple episodes for which we needed to save results
        # and we want to concatenate these episodes into one time series to get the overall result
        filepaths = results_dicty[key]
        # create empty list
        li = []
        # for each filepath, we read in the csv file as pandas dataframe and then append the df to the list
        for file in filepaths:
            df = pd.read_csv(file, index_col=0)
            li.append(df)
        # finally, we concatenate the df's in the list to one dataframe (concatenate on index axis
        # => below each other, since they build a time series)
        df = pd.concat(li)
        # rename the first column, which is always "datadate" in our results files
        df.rename(columns={df.columns[0]: "datadate"}, inplace = True)
        # rename the one other column for rewards and portfolio value using the respective key
        if key in ["pfvalue", "reward"]:
            df.rename(columns={df.columns[1]: key}, inplace = True)
        # for the state memory, we use the state header to as column names
        if key == "state_mem":
            df.columns = ["datadate"] + state_header_df.values.flatten().tolist()
        # sort based on date (since we want to have a nice time series and "glob" does not
        # necessarily import the fileüaths in the correct order)
        df = df.sort_values("datadate")
        # drop duplicate values (usually, the last state (where no action done anymore) is still saved in the episode results file,
        # and at the same time, it is saved in the results file of the next episode as the "initial" state, where we do an action.
        # This is not wrong (it is actuall practical to debug and check if the cirrect starting state is used in the episodes),
        # but we don't want to have it double here for time series analysis (wouldn't make sense))
        df = df.drop_duplicates(subset=["datadate"], keep='last')
        # include the results in the dictionary
        results_dicty.update({key: df})
    ### FOR BACKTESTING DICTIONARY
    backtest_dicty = backtest_dict.copy()
    for key in backtest_dicty:
        # for every key in results_dicty (e.g. "pfvalue", "reward",...) we get the list of filepaths
        # (there are multiple filepaths since we have multiple episodes for which we needed to save results
        # and we want to concatenate these episodes into one time series to get the overall result
        filepaths = backtest_dicty[key]
        # create empty list
        li = []
        # for each filepath, we read in the csv file as pandas dataframe and then append the df to the list
        for file in filepaths:
            df = pd.read_csv(file, index_col=0)
            li.append(df)
        # finally, we concatenate the df's in the list to one dataframe (concatenate on index axis
        # => below each other, since they build a time series)
        df = pd.concat(li)
        # rename the first column, which is always "datadate" in our results files
        df.rename(columns={df.columns[0]: "datadate"}, inplace = True)
        # rename the one other column for rewards and portfolio value using the respective key
        if key in ["pfvalue", "reward"]:
            df.rename(columns={df.columns[1]: key}, inplace = True)
        # for the state memory, we use the state header to as column names
        if key == "state_mem":
            df.columns = ["datadate"] + state_header_df.values.flatten().tolist()
        # sort based on date (since we want to have a nice time series and "glob" does not
        # necessarily import the fileüaths in the correct order)
        df = df.sort_values("datadate")
        # drop duplicate values (usually, the last state (where no action done anymore) is still saved in the episode results file,
        # and at the same time, it is saved in the results file of the next episode as the "initial" state, where we do an action.
        # This is not wrong (it is actuall practical to debug and check if the cirrect starting state is used in the episodes),
        # but we don't want to have it double here for time series analysis (wouldn't make sense))
        df = df.drop_duplicates(subset=["datadate"], keep='last')
        # include the results in the dictionary
        backtest_dicty.update({key: df})

    # the last three outputs are optional, just used for debugging
    return results_dicty, backtest_dicty,  state_header_df, results_dict, backtest_dict


def calculate_and_save_performance_metrics(results_dict: dict,
                                           backtest_dict: dict,
                                           save_path: str,
                                           seed: int = None,
                                           mode: str = "test"
                                           ) -> None:
    ### CALCULATE
    # sharpe ratio, max DD, average DD, total ret, USING RISK.FREE RATE OF 0
    # first, we need to convert datadate from integer to datetime format, so the library (ffn) an work with it
    results_dict["pfvalue"]["datadate"] = pd.to_datetime(results_dict["pfvalue"]["datadate"], format='%Y%m%d')
    backtest_dict["pfvalue"]["datadate"] = pd.to_datetime(backtest_dict["pfvalue"]["datadate"], format='%Y%m%d')
    # then we can create a "perf" object (performances) with the function .calc_stats() (bt = backtest)
    perf = results_dict["pfvalue"].set_index("datadate")["pfvalue"].calc_stats()
    bt_perf = backtest_dict["pfvalue"].set_index("datadate")["pfvalue"].calc_stats()
    # now we can access the statistics like this, for example: (ann = annuaized)
    sharpe_ratio_daily_ann = perf.daily_sharpe
    total_return = perf.total_return
    avg_daily_return_ann = perf.daily_mean
    std_daily_return_ann = perf.daily_vol
    maxdd = perf.max_drawdown
    avg_dd = perf.avg_drawdown
    avg_dd_days = perf.avg_drawdown_days
    # for backtest
    bt_sharpe_ratio_daily_ann = bt_perf.daily_sharpe
    bt_total_return = bt_perf.total_return
    bt_avg_daily_return_ann = bt_perf.daily_mean
    bt_std_daily_return_ann = bt_perf.daily_vol
    bt_maxdd = bt_perf.max_drawdown
    bt_avg_dd = bt_perf.avg_drawdown
    bt_avg_dd_days = bt_perf.avg_drawdown_days
    # calculate cumulative return
    #cumret = results_dict["pfvalue"]["pfvalue"]
    #bt_cumret = backtest_dict["pfvalue"]["pfvalue"]

    ### SAVE
    df = pd.DataFrame({"performance_metric":
                           ["sharpe_ratio_daily_ann", "total_return", "avg_daily_return_ann",
                            "std_daily_return_ann", "maxdd", "avg_dd", "avg_dd_days",
                            "bt_sharpe_ratio_daily_ann", "bt_total_return", "bt_avg_daily_return_ann",
                            "bt_std_daily_return_ann", "bt_maxdd", "bt_avg_dd", "bt_avg_dd_days"],
                       f"seed{seed}":
                           [sharpe_ratio_daily_ann, total_return, avg_daily_return_ann,
                            std_daily_return_ann, maxdd, avg_dd, avg_dd_days,
                            bt_sharpe_ratio_daily_ann, bt_total_return, bt_avg_daily_return_ann,
                            bt_std_daily_return_ann, bt_maxdd, bt_avg_dd, bt_avg_dd_days
                            ]
                       })
    df.to_csv(os.path.join(save_path, f"{mode}_performance_metrics_seed{seed}.csv"))
    return None