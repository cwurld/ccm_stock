from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pandas

from model import find_spikes, get_predictor_func_by_stock, calc_r_squared_for_list


class TestStockAnalysis(TestCase):

    # noinspection PyUnreachableCode
    def setUp(self) -> None:
        super().setUp()
        plt.rcParams["figure.figsize"] = (10, 4)
        n_days = 365
        self.time = np.arange(n_days)

        self.config = {
            'start_date': '20170103',
            'split_date': '20171231',
            'end_date': '20181231',
            'spike_length': 2,
            'spike_threshold_percent': 3.0,
            'predictor_func_length': 14,
            'price_field': 'Open',
            'predictor_field': 'Open'
        }

        # During the post covid bull market, the S&P gained 12% for the year
        self.target_slope = 15.0 / float(n_days)
        self.target_historical = self.time * self.target_slope + 100.0  # 115.00 at the end of the year

        self.predictor_slope = - 10.0 / float(n_days)
        self.predictor_historical = self.time * self.predictor_slope + 200.0

        self.spike = np.array([1.0, 1.03, 1.05, 1.04])  # fractional changes

        # The signal before the spike
        self.kernel = np.full(self.config['predictor_func_length'], 1.0)
        self.kernel[0] = 1.01
        self.kernel[3] = 0.97
        self.kernel[13] = 1.02

        # Add spikes to target historical data and pre-signal to predictor
        self.spike_times = [30, 100, 150]
        for i in self.spike_times:
            self.target_historical[i: i + self.spike.shape[0]] *= self.spike

            # Add pre-spike signal
            self.predictor_historical[i - self.kernel.shape[0]: i] *= self.kernel

        # Visually check
        if 0:
            plt.plot(self.predictor_historical)
            plt.plot(self.target_historical)
            for i in self.spike_times:
                plt.axvline(x=i, color='r')
            plt.show()

        self.data_sets = {}
        for symbol, data in [('TARGET', self.target_historical), ('PRED', self.predictor_historical)]:
            self.data_sets[symbol] = pandas.DataFrame.from_dict({'Open': data})

    def test_find_spikes(self):
        find_spikes(self.data_sets['TARGET'], self.config)

        spikes = self.data_sets['TARGET']['spikes'][self.data_sets['TARGET']['spikes']]
        self.assertEqual(self.spike_times, list(spikes.index))

        # print('done')

    def test_get_all_snippets(self):
        for symbol in ['TARGET', 'PRED']:
            find_spikes(self.data_sets[symbol], self.config)

        predictor_func_by_stock = get_predictor_func_by_stock(self.data_sets, 'TARGET', ['TARGET', 'PRED'], self.config)

        # kernel is not in target
        self.assertFalse(np.absolute(predictor_func_by_stock['TARGET'] - self.kernel).max() < 0.001)
        self.assertTrue(np.absolute(predictor_func_by_stock['PRED'] - self.kernel).max() < 0.001)

        # print('done')

    def test_prediction(self):
        for symbol in ['TARGET', 'PRED']:
            find_spikes(self.data_sets[symbol], self.config)

        predictor_func_by_stock = get_predictor_func_by_stock(self.data_sets, 'TARGET', ['TARGET', 'PRED'], self.config)

        # In these tests, TARGET does not have a predictor signal, so we will only use PRED
        calc_r_squared_for_list(self.data_sets, predictor_func_by_stock, self.config)

        df = self.data_sets['PRED']
        found_spikes = df['r_squared'][df['r_squared'] > 0.9].index.values
        self.assertEqual(self.spike_times, found_spikes.tolist())

        print('done')
