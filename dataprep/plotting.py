import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stockstats import StockDataFrame

# own libraries:
from config.config import *
from config.config import *
import logging

############################
##   SINGLE FUNCTIONS   ##
############################

def plot_historical_ts(df: pd.DataFrame,
                       subplot_rows: int = 2,
                       subplot_cols: int = 1,
                       datecol: str = "datadate",
                       comp_name_column: str = "tic",
                       comp_names_list: list = None, # []
                       x_cols_list: list = None, # [xcol plot 1, xcol plot 2], e.g. ["datadate", "datadate"]
                       y_cols_list: list = None, # [ycol plot 1, ycol plot 2], e.g. ["prccd", "log_prccd"]
                       x_labels_list: list = None, # [xlabel plot 1, xlabel plot 2]
                       y_labels_list: list = None, # [ylabel plot 1, ylabel plot 2]
                       titles_list: list = None,  # [title plot 1, title plot 2]
                       ylim_left_list: list = None,  # e.g. [0, -10]
                       ylim_right_list: list = None,  # e.g. [20000, 10]

                       # default figure / plot parameters
                       figsize: tuple = (17, 9),
                       space_between_plots: float = 0.5,
                       # default legend parameters
                       legend_location: str = "right",
                       legend_location_exact: tuple = (0.01, -0.04, 1.12, 1.1), # in relation to legend_location
                       # if legend_location = "right"; (right_loc, loc,up_loc, right_loc, up_loc) = (right_loc, up_loc)
                       legend_borderpad: float = 0.9,
                       legend_borderaxespad: float = 0.,
                       legend_fontsize: float = 16.,
                       # default label parameters
                       label_fontsize: float = 15.,
                       label_pad: float = 10.,
                       # default title parameters
                       title_fontsize: float = 18.,
                       title_pad: float = 18.,
                       ) -> None:
    df_ = df.copy()
    df_[datecol] = pd.to_datetime(df_[datecol], format='%Y%m%d')
    lines_list = []

    NUM_COLORS = len(comp_names_list)
    cm = plt.get_cmap('gist_rainbow')
    fig, axs = plt.subplots(nrows=subplot_rows, ncols=subplot_cols, sharex=False, sharey=False, figsize=figsize)
    if not isinstance(axs, list) and not isinstance(axs, np.ndarray):
        axs = (axs, np.NaN)

    for i in range(0, subplot_rows):
        x_col = x_cols_list[i]
        y_col = y_cols_list[i]
        title = titles_list[i]
        x_label = x_labels_list[i]
        y_label = y_labels_list[i]
        # xlim_left = xlim_left_list[i]
        # xlim_right = xlim_right_list[i]
        ylim_left = ylim_left_list[i]
        ylim_right = ylim_right_list[i]

        axs[i].set_prop_cycle('color', [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
        axs[i].set_title(title, fontsize=title_fontsize, pad=title_pad)
        axs[i].set_xlabel(x_label, fontsize=label_fontsize, labelpad=label_pad)
        axs[i].set_ylabel(y_label, fontsize=label_fontsize, labelpad=label_pad)
        # axs[i].set_xlim(xlim_left, xlim_right)
        # axs[i].ticklabel_format(useOffset=False, style="plain")
        if i == 0:
            from matplotlib.ticker import FuncFormatter
            axs[i].get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        axs[i].set_ylim(ylim_left, ylim_right)
        for comp_name in comp_names_list:
            subset = df_.loc[df_[comp_name_column] == comp_name]
            if i == 0:
                if not subset.empty:
                    try:
                        p = axs[i].plot(subset[x_col], subset[y_col])
                        lines_list.append(p)
                    except:
                        print(f"empty subset @ {comp_name}, i={i}")
                        pass
            else:
                if not subset.empty:
                    axs[i].plot(subset[x_col], subset[y_col])
    try:
        labs = comp_names_list[:len(lines_list) - 1]
        fig.legend(lines_list, labels=labs, loc=legend_location,
                   bbox_to_anchor=legend_location_exact,  # (x, y, width, height) 
                   borderaxespad=legend_borderaxespad,
                   borderpad=legend_borderpad, fontsize=legend_fontsize,
                   ncol=1)
        fig.tight_layout()
        fig.subplots_adjust(hspace=space_between_plots)
        plt.show()
    except:
        pass
    return None


def make_corr(df_pivoted: pd.DataFrame,
              columns_to_drop_for_calculation: list = None) -> None:
    corr = df_pivoted.drop(columns=columns_to_drop_for_calculation).corr(method='pearson')
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, cmap='BrBG_r', vmax=1.0, vmin=-1.0 , mask=mask, linewidths=2.5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()
    return None