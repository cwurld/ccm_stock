import math
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_spikes(df, df_column):
    df[df_column].plot(
        style='b+-',
        title='Opening Price for {} with {} Spikes'.format(df.attrs['stock_symbol'], df[df['spikes']].shape[0])
    )

    for xc in df[df['spikes']].index:
        plt.axvline(x=xc, color='deepskyblue', linestyle='-')


def plot_panel(symbols, title, plot_func, figsize=None, common_labels=None):
    """Generic code for plotting panel of sub-plots."""
    n_cols = 2
    n_rows = math.ceil(len(symbols) / n_cols)

    # noinspection PyTypeChecker
    if figsize:
        # noinspection PyTypeChecker
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, figsize=figsize)
    else:
        # noinspection PyTypeChecker
        fig, axs = plt.subplots(n_rows, n_cols, sharex=True)

    fig.tight_layout(pad=3.0)
    fig.suptitle(title, fontsize=16, y=0.99)
    for i, symbol in enumerate(symbols):
        row = i // n_cols
        col = i % n_cols
        plot_func(axs, row, col, symbol)
    plt.subplots_adjust(top=0.93)

    if common_labels:
        if len(axs.shape) == 1:
            plt.setp(axs[:], xlabel=common_labels[0])
            plt.setp(axs[0], ylabel=common_labels[1])
        else:
            plt.setp(axs[-1, :], xlabel=common_labels[0])
            plt.setp(axs[:, 0], ylabel=common_labels[1])


def plot_predictor_functions(symbols, target_stock, predictor_functions):
    def plot_func(axs, row, col, symbol):
        axs[row, col].plot(predictor_functions[symbol])
        axs[row, col].set_title(symbol)

    plot_panel(symbols, 'Predictor Functions for ' + target_stock, plot_func)


# ----------------------------------------------------------------------------------------------------------------
def plot_roc_base(df, the_plot):
    """Plots the ROC for one predictor stock."""
    the_plot.plot(df.fpr, df.tpr, 'b+-')
    line = np.arange(11) * 0.1
    the_plot.plot(line, line)


def plot_roc(auc_df, predictor_stock, title=None, data_set_name=None):
    """Plots the ROC for one predictor stock."""
    plot_roc_base(auc_df, plt)
    if title:
        plt.title(title)
    else:
        if data_set_name:
            plt.title('ROC for {} {}'.format(predictor_stock, data_set_name))
        else:
            plt.title('ROC for {}'.format(predictor_stock))


def plot_roc_panel(auc_df_by_predictor, target_stock, config):

    # Function for plotting in a subplot
    def plot_func(axs, row, col, symbol):
        df = auc_df_by_predictor[symbol]
        if len(axs.shape) == 1:
            the_panel = axs[col]
        else:
            the_panel = axs[row, col]

        plot_roc_base(df, the_panel)
        axs[row, col].set_title('{}: AUC={:.2f}'.format(symbol, df.attrs['auc']))

    plot_panel(
        auc_df_by_predictor.keys(),
        'ROC Curves for {} using {}'.format(target_stock, config['predictor_field']),
        plot_func,
        figsize=(10, 10),
        common_labels=['False Positive Rate', 'True Positive Rate']
    )


# ----------------------------------------------------------------------------------------------------------------
def plot_calc_performance_scores_by_threshold_base(df, the_plot):
    # For full plots and sub plots
    the_plot.errorbar(df['threshold'], df['exp_percent_change_mean'], yerr=df['exp_percent_change_std'],
                      fmt='-o', capsize=3)
    the_plot.grid()


def plot_calc_performance_scores_by_threshold(df, target_stock, predictor_stock):
    plot_calc_performance_scores_by_threshold_base(df, plt)
    plt.xlabel('Threshold')
    plt.ylabel('Expected Percent Change per Trade')
    plt.suptitle('Target: {}  Predictor: {}'.format(target_stock, predictor_stock))
    row = df[df.exp_percent_change_mean == df.exp_percent_change_mean.max()].iloc[0]
    plt.title('Max: {:.2f} at threshold: {:.2f}'.format(row.exp_percent_change_mean, row.threshold))


def plot_calc_performance_scores_by_threshold_for_predictors(target_stock, df_by_predictor, df_column):

    # Function for plotting in a subplot
    def plot_func(axs, row, col, symbol):
        df = df_by_predictor[symbol]
        if len(axs.shape) == 1:
            the_panel = axs[col]
        else:
            the_panel = axs[row, col]

        plot_calc_performance_scores_by_threshold_base(df, the_panel)
        best_row = df[df.exp_percent_change_mean == df.exp_percent_change_mean.max()].iloc[0]
        the_panel.set_title('{}: max={:.2f} at {:.2f}'.format(
            symbol, best_row.exp_percent_change_mean, best_row.threshold))

    plot_panel(df_by_predictor.keys(), 'Expected %Change/Trade for {} using {}'.format(target_stock, df_column),
               plot_func, figsize=(10, 10), common_labels=['Threshold of $r^2$', '% Change/Trade'])
