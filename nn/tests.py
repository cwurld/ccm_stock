from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from nn import NormalizeAndSplit


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
