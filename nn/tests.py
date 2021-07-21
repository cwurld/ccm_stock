from unittest import TestCase
import random

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from nn import NormalizeAndSplit
from nn2 import make_labels_for_threshold, make_labels
from window_generators import IntermixedWindowGenerator


class TestNormalizeAndSplit(TestCase):

    def setUp(self) -> None:
        super().setUp()
        data = {
            'd1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'd2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        }
        self.df = pd.DataFrame(data)
        self.splits = [0.5, 0.75]
        self.norm_split = NormalizeAndSplit(self.df, self.splits, 'd1')

    def test_norm_by_mean(self):
        df_train, df_val, df_test = self.norm_split.norm_by_mean()
        self.assertAlmostEqual(self.norm_split.train_mean['d1'], 3.0)
        self.assertAlmostEqual(self.norm_split.train_mean['d2'], 30.0)
        self.assertAlmostEqual(self.norm_split.train_std['d1'], 1.5811, places=2)
        self.assertAlmostEqual(self.norm_split.train_std['d2'], 15.811, places=2)

        # Use numpy tests because pandas assert_frame_equal does not handle indices correctly
        y = np.array([-1.26491, -0.63246, 0.00000, 0.63246, 1.26491])
        np.testing.assert_allclose(df_train['d1'], y, rtol=0.01)
        np.testing.assert_allclose(df_train['d2'], y, rtol=0.01)

        y = np.array([1.89737, 2.52982])
        np.testing.assert_allclose(df_val['d1'], y, rtol=0.01)
        np.testing.assert_allclose(df_val['d2'], y, rtol=0.01)

        y = np.array([3.16228, 3.79473, 4.42719])
        np.testing.assert_allclose(df_test['d1'], y, rtol=0.01)
        np.testing.assert_allclose(df_test['d2'], y, rtol=0.01)

        # De-normalize
        predictions = np.array([-1.26491, -0.63246, 0.00000, 0.63246, 1.26491, 1.89737,
                                2.52982, 3.16228, 3.79473, 4.42719])
        p2 = self.norm_split.de_normalize(predictions)
        np.testing.assert_allclose(p2, self.df['d1'].to_numpy(), rtol=0.01)

    def test_moving_avg(self):
        window_width = 2
        df_train, df_val, df_test = self.norm_split.moving_avg(window_width)
        expected_moving_average = {
            'd1': [np.nan, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
            'd2': [np.nan, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
        }
        assert_frame_equal(pd.DataFrame(expected_moving_average), self.norm_split.moving_avg_dataframe)

        # Use numpy tests because pandas assert_frame_equal does not handle indices correctly
        y = np.array([0.33333, 0.20000, 0.14286, 0.11111])
        np.testing.assert_allclose(df_train['d1'], y, rtol=0.01)
        np.testing.assert_allclose(df_train['d2'], y, rtol=0.01)

        y = np.array([0.09091, 0.07692])
        np.testing.assert_allclose(df_val['d1'], y, rtol=0.01)
        np.testing.assert_allclose(df_val['d2'], y, rtol=0.01)

        y = np.array([0.06667, 0.05882, 0.05263])
        np.testing.assert_allclose(df_test['d1'], y, rtol=0.01)
        np.testing.assert_allclose(df_test['d2'], y, rtol=0.01)

        # de-normalize
        predictions = \
            np.array([0.33333, 0.20000, 0.14286, 0.11111, 0.09091, 0.07692, 0.06667, 0.05882, 0.05263])
        p2 = self.norm_split.de_normalize(predictions)
        np.testing.assert_allclose(p2, self.df['d1'].to_numpy()[1:], rtol=0.01)

        predictions = np.array([0.33333, 0.20000, 0.14286, 0.11111])
        p2 = self.norm_split.de_normalize(predictions, 'train')
        np.testing.assert_allclose(p2, self.df['d1'][1:5].to_numpy(), rtol=0.01)

        predictions = np.array([0.09091, 0.07692])
        p2 = self.norm_split.de_normalize(predictions, 'val')
        np.testing.assert_allclose(p2, self.df['d1'][5:7].to_numpy(), rtol=0.01)

        # print('x')


def calc_n_pos(data):
    np = len([v for v in data if v > 0.5])
    nn = len(data) - np
    return np, nn


class TestIntermixedWindowGenerator(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.n = 100

        data = {
            'p1': np.arange(self.n, dtype=int),
            'p2': np.full((self.n,), 1, dtype=int),
            'p3': np.full((self.n,), 2, dtype=int)
        }
        self.df = pd.DataFrame(data)
        self.labels = np.arange(self.n, dtype=int)
        self.splits = [0.55, 0.82]
        self.window_width = 3
        self.min_run_length = 9

    def test_1(self):
        wg = IntermixedWindowGenerator(self.df, self.labels, self.splits, self.window_width, self.min_run_length)

        # Make sure the times in each set are unique and there are no lost time points
        times = {'train': set(), 'validation': set(), 'test': set()}
        all_times = set()
        for set_name in ['train', 'validation', 'test']:
            for sample, output_sample in zip(wg.inputs[set_name], wg.outputs[set_name]):
                times[set_name].update(sample[:, 0])  # all rows first column
                all_times.update(sample[:, 0], sample[:, 0])
                # Make sure the outputs align with the inputs
                self.assertEqual(sample[-1, 0], output_sample)  #

        self.assertFalse(times['train'].intersection(times['validation']))
        self.assertFalse(times['train'].intersection(times['test']))
        self.assertFalse(times['test'].intersection(times['validation']))
        self.assertEqual(set(list(range(99))), all_times)

        # print('x')

    def test_balanced(self):
        labels = np.zeros(self.n)

        # Make 30 positive
        random.seed(1)
        ii = random.sample(range(0, self.n), 30)
        labels[ii] = 1.0

        wg = IntermixedWindowGenerator(self.df, labels, self.splits, self.window_width, self.min_run_length,
                                       balanced=True)

        for set_name in ['train', 'validation']:
            n_pos1, n_neg1 = calc_n_pos(wg.balanced_outputs[set_name])
            n_pos2, n_neg2 = calc_n_pos(wg.outputs[set_name])
            self.assertEqual(n_pos1, n_pos2)
            self.assertTrue(abs(n_pos1 - n_neg1) < 2)

        # print('x')


class TestMakeLabels(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.threshold = 0.02
        self.conv_width = 3
        t = self.threshold + 0.01

        n = 50
        frac_diff = np.zeros(n)
        self.expected_labels = np.zeros(n)

        # Test edge effects by putting a block above threshold starting at 0.
        frac_diff[0:4] = t
        self.expected_labels[0] = 1.0

        # This increase occurs too close to prev, so it is ignored
        frac_diff[5] = t

        # Put in a double, only first one will be a label
        frac_diff[10:12] = t
        self.expected_labels[10] = 1.0

        # Put in a single
        frac_diff[20] = t
        self.expected_labels[20] = 1.0

        # Convert frac diff into prices
        self.prices = np.empty(n + 1)
        self.prices[0] = 100.0
        for i, fd in enumerate(frac_diff):
            self.prices[i + 1] = (1.0 + fd) * self.prices[i]

        self.prices = pd.DataFrame({'Open': self.prices})

    def test_1(self):
        labels = make_labels_for_threshold(self.prices['Open'], self.threshold, self.conv_width)
        np.testing.assert_almost_equal(self.expected_labels, labels)
        print('x')

    def test_2(self):
        labels, threshold, fp = make_labels(self.prices['Open'], self.conv_width, frac_positive=0.05)

        self.assertIsNotNone(labels)
        self.assertAlmostEqual(0.03, threshold)
        print('x')
