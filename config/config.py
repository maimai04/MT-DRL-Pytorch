import pathlib

#import finrl

import pandas as pd
import datetime
import os
#pd.options.display.max_rows = 10
#pd.options.display.max_columns = 10

#PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
#PACKAGE_ROOT = pathlib.Path().resolve().parent

#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
#DATASET_DIR = PACKAGE_ROOT / "data"

# data
#TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"
# BCAP: added strftime to avoid problem with ":" (Windows)
now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

# BCAP added which_run (later remove and use "now")
which_run = now

TRAINED_MODEL_DIR = f"trained_models\{now}"
os.makedirs(TRAINED_MODEL_DIR)

# BCAP added RESULTS_DIR as new directory for each run for results
RESULTS_DIR = f"results\{now}"
os.makedirs(RESULTS_DIR)

TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"


