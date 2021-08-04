import json
import os
import pickle
import time
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as stats

import leading_pattern.model as model
import my_utils.loaders as loaders

from leading_pattern.plotting import (
    plot_calc_performance_scores_by_threshold,
    plot_calc_performance_scores_by_threshold_for_predictors
)


def get_buy_indices(scores, threshold, config):
    """
    When to buy based on scores and threshold. Takes into account that for multiple successive scores above
    threshold, only the first will be a buy due to buy and hold strategy.
    """
    above_threshold = np.where(scores[0: -config['spike_length']] >= threshold)[0]
    buys = None
    if above_threshold.shape[0]:
        for i in above_threshold:
            if i >= config['predictor_func_length']:
                if buys is None:
                    buys = [i]

                # Fastest we can buy is every spike_length days
                for buy_i in above_threshold:
                    if buy_i - buys[-1] >= config['spike_length'] + 1:
                        buys.append(buy_i)
    else:
        buys = []
    return np.array(buys)  # the indices of buys


def calc_performance_for_buys(buys, prices, config):
    if buys.shape[0]:
        percent_change_per_buy = 100.0 * ((prices[buys + config['spike_length']] - prices[buys]) / prices[buys])

        # combines effects of price change with number of buys.
        gain = np.prod(prices[buys + config['spike_length']] / prices[buys])
    else:
        percent_change_per_buy = []
        gain = 0

    if len(percent_change_per_buy) > 0:
        percent_change_mean = np.mean(percent_change_per_buy)
    else:
        percent_change_mean = 0.0

    if len(percent_change_per_buy) > 1:
        percent_change_variance = np.var(percent_change_per_buy)
    else:
        percent_change_variance = 0.0

    return gain, percent_change_mean, percent_change_variance, percent_change_per_buy


def calc_performance_by_threshold(target_df, predictor_stock, config, min_threshold=0.1, data_set_name='test'):
    predictor_df = loaders.get_data_set(predictor_stock, config)
    predictor_func = model.get_predictor_func(target_df['train'], predictor_df['train'], config)

    model.calc_r_squared(predictor_df[data_set_name], predictor_func, config)
    prices = target_df[data_set_name][config['price_field']].values

    # The hypothesis is that the strategy picks buy times immediately before a price spike (e.g. not randomly). The
    # null hypothesis is that the strategy is randomly picking buy times. To calculate the probability that the null
    # hypothesis is true, we need to get the distribution of percent changes.
    #
    # One way to do this is to find out how many buys the strategy made and draw the same number of buys randomly
    # from the data. The problem with this approach is it combines that variance due to sampling with the actual
    # variance in the price data. If we had infinite data, we could just run the experiment over a lot of data
    # and the variance due to sampling would cancel out.
    #
    # Part of the reason this is not feasible is the data is not stationary. So we have to limit the date range
    # to make it approximately stationary.
    #
    # An alternative is to sample all changes in the data. This minimizes the variance from sampling.
    spike_length = config['spike_length']
    null_percent_changes = \
        100.00 * ((prices[spike_length:] - np.roll(prices, spike_length)[spike_length:]) /
                  np.roll(prices, spike_length)[spike_length:])
    null_hypothesis_stats = {'mean': np.mean(null_percent_changes), 'std': np.std(null_percent_changes)}

    performance = []
    threshold = min_threshold
    while threshold <= 0.9:
        r = {
            'threshold': threshold,
            'n_trades': 0,
            'percent_change_mean': 0.0,
            'percent_change_std': 0.0,
            'pvalue': 1.0,
            'gain': 1.0,
            'score': 0.0
        }

        buys = get_buy_indices(predictor_df[data_set_name].r_squared.values, threshold, config)
        if buys.shape[0] > 5:
            # noinspection PyUnresolvedReferences
            r['n_trades'] = int(buys.shape[0])

            # Calculate the percent change for each buy
            r['gain'], r['percent_change_mean'], variance, percent_change_per_buy = \
                calc_performance_for_buys(buys, prices, config)
            r['percent_change_std'] = math.sqrt(variance)

            kolmogorov_smirnov = stats.ks_2samp(null_percent_changes, percent_change_per_buy, alternative='greater')
            # noinspection PyUnresolvedReferences
            r['pvalue'] = kolmogorov_smirnov.pvalue

            # For some predictors the null hypothesis will be true. For the buys are not entirely random. We don't
            # know which and actually do not care as long as the price is going up. Expected percent change
            # is the change for each hypothesis weighted by the probability that hypothesis is true.
            w1 = (1.0 - r['pvalue'])
            w2 = r['pvalue']
            r['exp_percent_change_mean'] = w1 * r['percent_change_mean'] + w2 * null_hypothesis_stats['mean']
            r['exp_percent_change_std'] = \
                math.sqrt((w1 * r['percent_change_std']) ** 2.0 + (w2 * null_hypothesis_stats['std']) ** 2.0)
            performance.append(r)

        threshold += 0.05

    if performance:
        df = pd.DataFrame(performance)
        df.set_index('threshold')
        df.attrs['null_hypothesis_stats'] = null_hypothesis_stats
    else:
        df = None
    return df


def calc_performance_by_threshold_for_predictors(
        target_stock, predictor_stocks, config, data_set_name='test', min_threshold=0.1):
    """Runs calc_performance_by_threshold() on a list of predictor stock symbols."""
    target_df = loaders.get_data_set(target_stock, config)
    df_by_predictor = {}
    for predictor_stock in predictor_stocks:
        df_by_predictor[predictor_stock] = \
            calc_performance_by_threshold(
                target_df, predictor_stock, config, data_set_name=data_set_name, min_threshold=min_threshold)
    return df_by_predictor


def compare_train_test(stock, config, n_rows=10):
    results = {'predictor': [], 'train': [], 'test': []}
    yield_data = {}
    for data_set_name in ['train', 'test']:
        filename_suffix = '_'.join([stock, config['predictor_field'], data_set_name])
        pickle_path = model.get_results_dir(config) / 'predictors_{}.pickle'.format(filename_suffix)
        with open(pickle_path, 'rb') as fp:
            yield_data[data_set_name] = pickle.load(fp)

    # pickle_data[predictor_symbol] = [threshold, best.exp_percent_change_mean]
    performance_list = sorted([[k, v] for k, v in yield_data['train'].items()], key=lambda x: x[1][1], reverse=True)

    count = 0
    for symbol, v in performance_list:
        results['predictor'].append(symbol)
        results['train'].append(v[1])
        results['test'].append(yield_data['test'][symbol][1])
        if count >= n_rows:
            break
        else:
            count += 1

    df = pd.DataFrame(results)
    df.set_index('predictor')
    return df


def plot_train_vs_test(target_stock, predictor_stock, config):
    data_sets = loaders.get_data_sets([target_stock, predictor_stock], config)

    model.find_spikes(data_sets['train'][target_stock], config)
    predictor_func = model.get_predictor_snippets(
        data_sets['train'][target_stock], data_sets['train'][predictor_stock], config)[0]

    model.calc_r_squared(data_sets['train'][predictor_stock], predictor_func, config)
    model.calc_r_squared(data_sets['test'][predictor_stock], predictor_func, config)


def run_all_predictors(target_stock, config, data_set_name='test', dry_run=False):
    """
    Calculate performance for all predictor stocks, for the target stock. Write a dict of results by predictor stock
    to a pickle file. Write a summary of the performance values to a summary json file.
    """
    np.seterr(all='raise')  # make code stop on warnings
    filename_suffix = '_'.join([target_stock, config['predictor_field'], data_set_name])
    results_dir = model.get_results_dir(config)

    target_df = loaders.get_data_set(target_stock, config)
    model.find_spikes(target_df['train'], config)

    if data_set_name == 'test':
        model.find_spikes(target_df['test'], config)
        n_test = model.count_spikes(target_df['test'])
    else:
        n_test = 0

    # Do not run the analysis if there are less than 10 spikes to train on
    n_train = model.count_spikes(target_df['train'])
    if n_train < 10:
        stats = {
            'data_set_name': data_set_name,
            'n_train_spikes': n_train,
            'n_test_spikes': n_test,
            'performance': [],
        }

        with open(results_dir / 'summary_{}.json'.format(target_stock, filename_suffix), 'w') as fp:
            json.dump(stats, fp, indent=4)
        return

    # Write intermediate results to file in case there is a crash. Load the intermediate results if they exist
    performance_by_stock = {}
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    out_path = results_dir / 'predictors_{}.pickle'.format(filename_suffix)
    if os.path.exists(out_path):
        with open(out_path, 'rb') as fp:
            performance_by_stock = pickle.load(fp)

    count = 0
    start = time.time()
    predictor_symbols = loaders.get_all_symbols_in_cache(config)
    for i, predictor_symbol in enumerate(predictor_symbols):
        if predictor_symbol not in performance_by_stock:
            df = calc_performance_by_threshold(target_df, predictor_symbol, config, min_threshold=0.5,
                                               data_set_name=data_set_name)

            if df is None:
                performance_by_stock[predictor_symbol] = [0.0, 0.0]
            else:
                best_row = df[df.exp_percent_change_mean == df.exp_percent_change_mean.max()].iloc[0]
                performance_by_stock[predictor_symbol] = [best_row.threshold, best_row.exp_percent_change_mean]

            # Save intermediate data and print some results to show progress
            count += 1
            if count > 100:
                with open(out_path, 'wb') as fp:
                    pickle.dump(performance_by_stock, fp)
                count = 0
                now = time.time()
                print('Completed: {}  {}  {:.2f} seconds'.format(i, target_stock, now - start))
                start = now

            if dry_run and i > 200:
                break

        with open(out_path, 'wb') as fp:
            pickle.dump(performance_by_stock, fp)

    calc_summary_stats(target_df, config, n_train=n_train, n_test=n_test, data_set_name=data_set_name)


def calc_summary_stats(target_stock_df, config, n_train=None, n_test=None, data_set_name='test'):
    df = target_stock_df['train']
    symbol = df.attrs['stock_symbol']
    filename_suffix = '_'.join([symbol, config['predictor_field'], data_set_name])

    pickle_path = model.get_results_dir(config) / 'predictors_{}.pickle'.format(filename_suffix)
    with open(pickle_path, 'rb') as fp:
        performance_by_stock = pickle.load(fp)

    if n_train is None:
        model.find_spikes(target_stock_df['train'], config)
        n_train = model.count_spikes(target_stock_df['train'])

    if n_test is None:
        model.find_spikes(target_stock_df['test'], config)
        n_test = target_stock_df['test']['spikes'][target_stock_df['test']['spikes']].shape[0]

    # Write the top 20 AUC in descending order
    stats = {
        'data_set_name': data_set_name,
        'n_train_spikes': n_train,
        'n_test_spikes': n_test,
        'symbol_performance_list':
            sorted([(k, v) for k, v in performance_by_stock.items()], key=lambda x: x[1][1], reverse=True)[0:20],
    }

    # Write as JSON so it is human readable and available for post processing
    with open(model.get_results_dir(config) / 'summary_{}.json'.format(filename_suffix), 'w') as fp:
        json.dump(stats, fp, indent=4)


def plot_run_all_predictors_results(target_stock, config, n_predictors=10):

    # Get the best predictor stocks from all the possible predictor stocks from the summary file
    results_path = model.get_results_dir(config) / 'summary_{}.json'.format(target_stock)
    with open(results_path, 'r') as fp:
        results = json.load(fp)

    # Load data and get predictor functions from the training data
    predictor_stocks = [x[0] for x in results['symbol_performance_list'][0:n_predictors]]
    df_by_predictor = calc_performance_by_threshold_for_predictors(target_stock, predictor_stocks, config)
    plot_calc_performance_scores_by_threshold_for_predictors(target_stock, df_by_predictor, config['predictor_field'])


def skunk():
    # Keeps variables out of global name-space
    config = {
        'start_date': '20170103',
        'split_date': '20171231',
        'end_date': '20181231',
        'spike_length': 2,
        'spike_threshold_percent': 3.0,
        'predictor_func_length': 7,
        'use_diffs': True,  # when True the predictor functions are a times series of diffs
        'price_field': 'Open',
        'predictor_field': 'Open'
    }

    func_names = [
        # 'calc_performance_by_threshold',
        # 'calc_performance_by_threshold_for_predictors',
        'run_all_predictors',
        # 'make_summaries',
        # 'plot_run_all',
        # 'compare_train_test'
    ]

    if 'calc_performance_by_threshold' in func_names:
        # config['predictor_field'] = 'Volume'
        target_stock = 'NPTN'
        predictor_stock = 'NOMD'
        target_df = loaders.get_data_set(target_stock, config)

        df = calc_performance_by_threshold(target_df, predictor_stock, config, data_set_name='test')
        plot_calc_performance_scores_by_threshold(df, target_stock, predictor_stock)
        plt.show()

        df = calc_performance_by_threshold(target_df, predictor_stock, config, data_set_name='train')
        plot_calc_performance_scores_by_threshold(df, target_stock, predictor_stock)

        plt.show()
        print('x')

    if 'calc_performance_by_threshold_for_predictors' in func_names:
        # Look for predictors in a list of target stocks

        # noinspection SpellCheckingInspection
        target_stock = 'NPTN'
        predictor_stocks = ['PHD']  # ['NPTN', 'MTL', 'AGR', 'PHD', 'ESS', 'CBOE', 'AGRO', 'FPF', 'CHMA', 'SEAS']
        df_by_predictor = calc_performance_by_threshold_for_predictors(target_stock, predictor_stocks, config)
        plot_calc_performance_scores_by_threshold_for_predictors(
            target_stock, df_by_predictor, config['predictor_field'])
        plt.show()
        return df_by_predictor

    if 'plot_train_vs_test' in func_names:
        pass

    if 'run_all_predictors' in func_names:
        target_stock = 'NPTN'
        run_all_predictors(target_stock, config, data_set_name='test', dry_run=False)
        run_all_predictors(target_stock, config, data_set_name='train', dry_run=False)
        print('done')

    if 'make_summaries' in func_names:
        # Update the summary files

        # noinspection SpellCheckingInspection
        symbols = ['MTL']  # ['OTIC', 'CFRX', 'TANH', 'CCM', 'LUNA', 'RBCAA', 'ARLO', 'DSWL', 'SLAB', 'OHI']
        for symbol in symbols:
            df = loaders.get_data_set(symbol, config)
            calc_summary_stats(df, config, data_set_name='train')

    if 'plot_run_all' in func_names:
        # noinspection SpellCheckingInspection
        plot_run_all_predictors_results('NPTN', config)
        plt.show()

    if 'compare_train_test' in func_names:
        stock = 'NPTN'
        df = compare_train_test(stock, config)
        print(df)


if __name__ == '__main__':
    r = skunk()
