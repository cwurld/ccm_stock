import datetime
import os
from pathlib import Path
import csv

import numpy as np
import pandas_datareader.data as web
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / 'data'


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
