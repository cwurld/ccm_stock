import datetime
import json
import os
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web

import matplotlib.pyplot as plt
import scipy.stats as stats

from plotting import plot_spikes, plot_predictor_functions


DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'results'


def get_results_dir(config):
    return RESULTS_DIR / '{}_{}'.format(config['start_date'], config['end_date'])


def read_nasdaq_stock_list(raw=True):
    # From: https://www.nasdaq.com/market-activity/stocks/screener
    df = pd.read_csv(DATA_DIR / 'nasdaq_screener_1619356287441.csv', header=0)
    if raw:
        return df

    # "Last Sale" is a string because each element starts with $. Remove it and convert to float
    df['Last Sale'] = df['Last Sale'].apply(lambda x: float(x.replace('$', '')))

    # Add a column of dollar amount of last sale times volume
    df['dollar_vol'] = df['Last Sale'] * df['Volume']
    return df


def select_random_stock_symbols(
        nasdaq_stock_list_df, n_symbols, analysis_start_year, quantile=0.25,
        filter_field='dollar_vol', seed=1):
    """
    Randomly selects n stocks in the above the quantile of the filter_field.
    """
    # Get value of the quantile
    q1 = nasdaq_stock_list_df[filter_field].quantile([quantile]).iat[0]

    # Randomly select n stocks with a market cap above the quantile
    df2 = nasdaq_stock_list_df[
        (nasdaq_stock_list_df[filter_field] > q1) &
        (nasdaq_stock_list_df['IPO Year'] < analysis_start_year - 1.0)
        ]

    stock_symbols = df2.sample(n=n_symbols, random_state=seed)
    return list(stock_symbols['Symbol'].values)


def get_stock_historical_data(symbol, config):
    """
    Get stock historical data for a single stock. Try the cache first.
    If it's not in there, get it from Yahoo and cache it.
    """
    start_date_dt = datetime.datetime.strptime(config['start_date'], '%Y%m%d').date()
    end_date_dt = datetime.datetime.strptime(config['end_date'], '%Y%m%d').date()

    filename = '{}.csv'.format(symbol)
    dir_path = DATA_DIR / 'nasdaq_historical' / '{}_{}'.format(config['start_date'], config['end_date'])
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    full_path = dir_path / filename
    if os.path.exists(full_path):
        df = pd.read_csv(full_path, header=0, index_col='Date', parse_dates=True)
    else:
        df = web.get_data_yahoo(symbol, config['start_date'], config['end_date'])
        if df.index[0].date() == start_date_dt and df.index[-1].date() == end_date_dt:
            df.to_csv(full_path)

    df.attrs.update(config)
    df.attrs['stock_symbol'] = symbol
    df.attrs['start_date_dt'] = datetime.datetime.strptime(config['start_date'], '%Y%m%d').date()
    df.attrs['end_date_dt'] = datetime.datetime.strptime(config['end_date'], '%Y%m%d').date()
    df.attrs['date_dir'] = '{}_{}'.format(config['start_date'], config['end_date'])  # for building paths
    return df


def fetch_all_stocks(config):
    """
    Get historical data from Yahoo for all stocks and write as CSV to data folder. Skip if we already have the
    stock.
    """
    start_date_dt = datetime.datetime.strptime(config['start_date'], '%Y%m%d').date()
    end_date_dt = datetime.datetime.strptime(config['end_date'], '%Y%m%d').date()

    with open(DATA_DIR / 'nasdaq_screener_1619356287441.csv', 'r') as fp:
        reader = csv.reader(fp)
        next(reader)  # skip header
        symbols = [row[0] for row in reader]

    dir_path = DATA_DIR / 'nasdaq_historical' / '{}_{}'.format(config['start_date'], config['end_date'])
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for symbol in symbols:
        filename = '{}.csv'.format(symbol)
        full_path = dir_path / filename
        if not os.path.exists(full_path):
            try:
                df = web.get_data_yahoo(symbol, config['start_date'], config['end_date'])
            except:
                # print(traceback.format_exc())
                print('Could not load ' + symbol)
            else:
                if df.index[0].date() == start_date_dt and df.index[-1].date() == end_date_dt:
                    print('Loaded: ' + symbol)
                    df.to_csv(full_path)


def get_data_set(symbol, config):
    """
    Gets primary data for symbol.
    """
    try:
        df = get_stock_historical_data(symbol, config)
    except:
        df = None
    else:
        # Makes it easier to use Volume as an indicator, like price
        df['Volume'] = df['Volume'].astype(np.float64)

    if config.get('split_date'):
        return {
            'train': df[config['start_date']:config['split_date']],
            'test': df[config['split_date']:config['end_date']]
        }
    else:
        return df


def get_data_sets(symbols, config):
    """
    Gets primary data for each symbol. If there is a split date, then the results are packaged to make it easier to
    get either all training data or all testing data.
    """
    if config.get('split_date'):
        data_sets = {'train': {}, 'test': {}}
    else:
        data_sets = {}

    for symbol in symbols:
        if config.get('split_date'):
            data_set = get_data_set(symbol, config)
            data_sets['train'][symbol] = data_set['train']
            data_sets['test'][symbol] = data_set['test']
        else:
            data_sets[symbol] = get_data_set(symbol, config)
    return data_sets


def get_all_symbols_in_cache(config):
    data_dir = DATA_DIR / 'nasdaq_historical' / '{}_{}'.format(config['start_date'], config['end_date'])
    symbols = [f_name.split('.')[0] for f_name in os.listdir(data_dir)]
    return symbols


# Spikes ----------------------------------------------------------------------------------------------------------
def find_spikes(df, config):
    """
    Finds price spikes. This is the day we would "buy".

    If a[i] is the background value, and a[i+1], ... a[i+spike_length] are monotonically increasing and
    100 * (a[i+spike_length] - a[i])/a[1] > threshold, then spikes[i] is set to True.

    When buying based on this algorithm, we would buy at the open of day i and sell at the open of i+spike_length.

    Note that a n day spike length involves n+1 prices.
    """
    threshold_fraction = config['spike_threshold_percent'] / 100.0
    diffs = df[config['price_field']].diff().fillna(0.0).div(df[config['price_field']])

    # Continue looking for pattern starting where one-day diff exceeds the one-day threshold, continue until
    # diff is negative
    spikes = [False] * df.shape[0]

    n_positive = 0
    change = 0.0
    for i, d in enumerate(diffs):
        if d > 0.0:
            n_positive += 1
            change += d

            if n_positive >= config['spike_length']:
                if change > threshold_fraction:
                    spikes[i - config['spike_length']] = True
                n_positive = 0
                change = 0.0
        else:
            n_positive = 0
            change = 0.0

    df['spikes'] = spikes  # return result as a new column


def count_spikes(df):
    return df['spikes'][df['spikes']].shape[0]


def find_all_spikes(config):
    """Writes a file with counts of train and test spikes by stock symbol"""
    symbols = get_all_symbols_in_cache(config)
    spikes = []
    for symbol in symbols:
        df = get_data_set(symbol, config)
        n_spikes = [symbol]
        for ds_name in ['train', 'test']:
            find_spikes(df[ds_name], config)
            n_spikes.append(df[ds_name].spikes[df[ds_name].spikes].shape[0])
        spikes.append(n_spikes)
    spikes.sort(key=lambda x: x[1], reverse=True)  # sort by train

    with open(get_results_dir(config) / 'spikes.json', 'w') as fp:
        json.dump(spikes, fp, indent=4)

    return spikes  # list of [symbol, n train, n test]


# Predictor functions ----------------------------------------------------------------------------------------------
def get_predictor_snippets(target_df, predictor_df, config):
    if not hasattr(target_df, 'spikes'):
        find_spikes(target_df, config)

    spikes = np.where(target_df.spikes)[0]
    time_series = np.copy(predictor_df[config['predictor_field']].to_numpy(np.float64))
    if config['use_diffs']:
        time_series = np.diff(time_series, prepend=time_series[0:1])

    snippets = []
    avg_snippet = np.zeros(config['predictor_func_length'])

    n_snippets = 0.0
    for spike in spikes:
        if spike > config['predictor_func_length']:  # avoid edge effects
            snippet = time_series[spike - config['predictor_func_length']: spike]
            snippet_avg = snippet.sum() / config['predictor_func_length']

            if abs(snippet_avg) > 0.001:
                snippet /= snippet_avg

                if snippet.shape == avg_snippet.shape:
                    avg_snippet += snippet
                    snippets.append(snippet)
                    n_snippets += 1.0

    if n_snippets > 0.1:
        avg_snippet /= n_snippets

    return avg_snippet, snippets


def get_predictor_func(target_df, predictor_df, config):
    return get_predictor_snippets(target_df, predictor_df, config)[0]


def get_predictor_func_by_stock(data_sets, target_stock, predictor_stock_symbols, config):
    predictor_func_by_stock = {}
    for predictor_stock in predictor_stock_symbols:
        predictor_func_by_stock[predictor_stock] = \
            get_predictor_func(data_sets[target_stock], data_sets[predictor_stock], config)
    return predictor_func_by_stock


# r^2 Scores ------------------------------------------------------------------------------------------------------
def calc_r_squared(predictor_df, predictor_func, config):
    """
    At each point in the predictor historical data, get a snippet the length of the predictor function and try to
    fit snippet vs predictor function by a line and record the r^2-value.
    """
    k_len = predictor_func.shape[0]

    historical_data = predictor_df[config['predictor_field']].to_numpy()
    if config['use_diffs']:
        historical_data = np.diff(historical_data, prepend=historical_data[0:1])

    r_squared = np.zeros_like(historical_data)
    for shift in range(0, historical_data.shape[0] - k_len):
        snippet = historical_data[shift: k_len + shift]
        try:
            fit = stats.linregress(snippet, predictor_func)
        except:
            r_squared[shift + k_len] = 0.0  # to handle vertical lines
        else:
            r_squared[shift + k_len] = fit.rvalue * fit.rvalue
    predictor_df['r_squared'] = r_squared


def calc_r_squared_for_list(data_sets, predictor_func_by_stock, config):
    """Add scores column to predictor data_sets"""
    for symbol, predictor_func in predictor_func_by_stock.items():
        calc_r_squared(data_sets[symbol], predictor_func, config)


def skunk():
    # Keeps variables out of global name-space
    config = {
        'start_date': '20170103',
        'split_date': '20171231',
        'end_date': '20181231',
        'spike_length': 2,
        'spike_threshold_percent': 3.0,
        'predictor_func_length': 14,
        'use_diffs': True,  # when True the predictor functions are a times series of diffs
        'price_field': 'Open',
        'predictor_field': 'Open'
    }

    func_names = [
        # 'fetch_one',
        # 'fetch_all',
        # 'get_one'
        'find_all_spikes',
        # 'plot_spikes',
        # 'plot_predictor_funcs'
    ]

    if 'fetch_one' in func_names:
        df = get_stock_historical_data('OTIC', config)
        print(df.info())

    if 'fetch_all' in func_names:
        fetch_all_stocks(config)

    if 'get_one' in func_names:
        df = get_data_set('CCM', config)
        print(df['train'].info())
        print(df['test'].info())

    if 'find_all_spikes' in func_names:
        spikes = find_all_spikes(config)
        return spikes

    if 'plot_spikes' in func_names:
        target = 'PLSE'
        df = get_data_set(target, config)
        find_spikes(df['train'], config)  # Find spikes in target stock and a column 'spikes' to dataframe.
        plot_spikes(df['train'], config['price_field'])
        plt.show()

        find_spikes(df['test'], config)  # Find spikes in target stock and a column 'spikes' to dataframe.
        plot_spikes(df['test'], config['price_field'])
        plt.show()
        print('done')

    if 'get_snippets' in func_names:
        target_df = get_data_set('CCM', config)
        predictor_df = get_data_set('OTIC', config)
        find_spikes(target_df['train'], config)
        get_predictor_snippets(target_df['train'], predictor_df['train'], config)

    if 'plot_predictor_funcs' in func_names:
        predictor_stocks = ['NPTN', 'MTL', 'AGR', 'PHD', 'ESS', 'CBOE', 'AGRO', 'FPF', 'CHMA', 'SEAS']
        target_stock = 'NPTN'
        data_sets = get_data_sets(predictor_stocks, config)
        df = data_sets['train'][target_stock]
        find_spikes(df, config)
        predictor_func_by_stock = \
            get_predictor_func_by_stock(data_sets['train'], target_stock, predictor_stocks, config)

        plot_predictor_functions(predictor_stocks, target_stock, predictor_func_by_stock)
        plt.show()
        print('done')


if __name__ == '__main__':
    r = skunk()
