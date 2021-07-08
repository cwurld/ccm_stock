import json
from pathlib import Path
from unittest import TestCase

import numpy as np

import my_utils.loaders as loaders


def volatility(time_series):
    frac_diff = np.diff(time_series, prepend=[time_series[0]]) / time_series
    q = np.quantile(frac_diff, [0.25, 0.75])
    return q[1] - q[0]


def all_volatility(results_dir, config):
    symbols = loaders.get_all_symbols_in_cache(config)
    results = []
    for symbol in symbols:
        df = loaders.get_data_set(symbol, config)
        if (df['Open'].max() < 1000.0) and (df['Open'].min() > 1.0):
            v = volatility(df['Open'].to_numpy())
            results.append((symbol, v))
    results.sort(key=lambda x: x[1], reverse=True)

    with open(results_dir / 'volatility.json', 'w') as fp:
        json.dump(results, fp, indent=4)


def load_volatility(results_dir, config):
    with open(results_dir / 'volatility.json', 'r') as fp:
        v = json.load(fp)
    return v


if __name__ == '__main__':
    config = {
        'start_date': '20170103',
        'end_date': '20181231'
    }
    results_dir = Path(__file__).parent.parent / 'nn' / 'results' / '20170103_20181231'
    all_volatility(results_dir, config)


class TestVolatility(TestCase):
    def test_1(self):
        ts = np.array([1, 2, 3, 4])
        v = volatility(ts)
        # noinspection PyTypeChecker
        self.assertAlmostEqual(0.1039, v, 3)
