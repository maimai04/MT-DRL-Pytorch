import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from config.config import *
from preprocessing.preprocessors import *
import re

#get path for each seed within run_of_choice
def get_all_fs(f_path, string_condition="*Seed*"):
    """
    function using glob package to get all folders or files paths within a directory based on a string condition.
    @param f_path           : location (path) of the folders / files
    @param string_condition : used to identify the files/ folders to collect into the list,
                              e.g. "seed(.*)_model" => all folders names "seed1_model, seed2_model etc.
    @return: list of all collected file/ folder paths we were looking for
    """
    all_fs_list = glob.glob(os.path.join(f_path, string_condition))
    return all_fs_list

def get_substring(string, string_condition="words(.*)words", group=1):
    """
    function to get a part of a string based on a condition.
    """
    result = re.search(string_condition, string)
    result = result.group(group)
    return result

def create_run_dict(all_seed_folder_paths, seed_string_condition=f"agent(.*)"):
    run_dict = {}
    for seed_folder in all_seed_folder_paths:
        # get seed name to be used as key within main dictionary (level 1 key)
        key_seed = get_substring(string=seed_folder, string_condition=seed_string_condition)
        run_dict.update({key_seed: {}})

        all_outcomes_folders_list = get_all_fs(f_path=seed_folder, string_condition="*")

        for outcome_folder in all_outcomes_folders_list:
            outcome_folder_name = os.path.basename(outcome_folder) # get name of folder without path
            #print(outcome_folder)
            key_outcome = outcome_folder_name
            run_dict[key_seed].update({key_outcome: {}})

            # get al training, validation, trade files
            for i in ["train", "validation", "trade"]:
                all_files_list = get_all_fs(f_path=outcome_folder, string_condition=f"*{i}*")
                #if i == "validation": print(all_files)
                # for now, we only store the paths, so we don't use up memory for loaded data we mught not actually ned atm
                # as soon as we need data, we import it into the respective dict part
                run_dict[key_seed][key_outcome].update({i: all_files_list})

    return run_dict

def concat_csv_to_df(file_paths_list,
                     axis=0,
                     sorting=True,
                     sorting_condition="_i(.*).csv"
                     ) -> list:
    dflist = []
    # sort files based on iteration number, so that they are in the correct order for concatenation to pd.DataFrame
    if sorting:
        file_paths_list = sorted(file_paths_list, key=lambda x: int(re.search(sorting_condition, x).group(1)))
    for filepath in file_paths_list:
        df = pd.read_csv(filepath, index_col=0)
        dflist.append(df)
    df = pd.concat(dflist, axis=axis, ignore_index=True)
    return [df, file_paths_list]

def load_combine_results(dictionary,
                         load_list,
                         mode="trade",
                         combine_mode="within_seed",
                         sorting=True,
                         sorting_condition="_i(.*).csv"
                         ) -> list:
    """
    load results based on paths given.
    Combine results csv files for one seed, one mode (e.g. trade) in one df.
    Save the new df where the paths list was located before in the dictionary (overwriting file paths list).
    @param dictionary:
    @param load_list: list of names like ["portfolio_value", "cash_value",...]
    @param mode:
    @param combine_mode:
    @param sorting:
    @param sorting_condition:
    @return:
    """
    from functools import reduce

    dict_ = dictionary.copy()
    if combine_mode == "within_seed":
        for seed_key in dict_:
            for result_elem in load_list:

                files_paths_list = dict_[seed_key][result_elem][mode]
                if len(files_paths_list) > 1:
                    df_conc, file_paths_list = concat_csv_to_df(file_paths_list=files_paths_list,
                                                                       axis=0, sorting=sorting, # must be True
                                                                       sorting_condition=sorting_condition)
                    dict_[seed_key][result_elem][mode] = df_conc
        return [dict_, file_paths_list]

    if combine_mode == "across_seeds":
        for result_elem in load_list:
            print(result_elem)
            temp_list = []
            for seed_key in dict_:
                # print(seed_key)
                seed_df = dict_[seed_key][result_elem][mode]
                seed_df.rename(columns={result_elem: result_elem + "_" + seed_key}, inplace=True)
                # seed_df.set_index("datadate", inplace=True)
                temp_list.append(seed_df)
                # df_conc = pd.concat(temp_list, axis=1, ignore_index=True)
            df_conc = reduce(lambda left, right: pd.merge(left, right, on='datadate'), temp_list)
            dict_.update({result_elem: df_conc})
        return [dict_, None]