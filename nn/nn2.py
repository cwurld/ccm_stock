# Binary Classification
from pathlib import Path

import numpy as np

import tensorflow as tf

RESULTS_DIR = Path(__file__).parent / 'results'

CONV_WIDTH = 3


def get_results_dir(config):
    return RESULTS_DIR / '{}_{}'.format(config['start_date'], config['end_date'])


def diff_norm(df):
    norm_df = df.copy()
    norm_df = (norm_df.diff(1) / norm_df).iloc[1:, :]
    norm_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return norm_df


def make_labels(prices, conv_width, frac_positive=0.2):
    """Loop through thresholds to find one that produces the specified frac positive."""
    threshold = 0.05
    fp = 0.0
    while threshold > 0.0:
        labels = make_labels_for_threshold(prices, threshold, conv_width)
        ii = np.where(labels > 0.5)
        fp = ii[0].shape[0] / prices.shape[0]
        if fp > frac_positive:
            return labels, threshold, fp
        threshold -= 0.005

    return None, threshold, fp


def make_labels_for_threshold(prices, threshold, conv_width):
    """
    Our hypothesis is that an event triggers a price spike and we are trying to detect the start of that spike.

    Our model is not designed to handle multi-day increases. So simply labelling that data by price changes that
    are above threshold is not correct.

    This function only labels the first price change that is above threshold in a series. It also only
    accepts first price changes that are preceded by conv_window price that are below the threshold.
    """
    frac_diff = ((-prices.diff(-1).to_numpy()) / prices)[0: -1]
    labels = np.zeros(frac_diff.shape[0])
    black_out = -1
    for i, x in enumerate(frac_diff):
        if x >= threshold:
            if i > black_out:
                # If there are multiple successive prices above threshold, take the first one
                if i == 0 or frac_diff[i - 1] < threshold:
                    labels[i] = 1.0
            black_out = i + conv_width
    return labels


def add_signals(df, labels, kernels):
    """
    For testing models.

    Adds signals to predictors before each positive label. This should make the models have an accuracy near 100%.
    """
    for i, label in enumerate(labels):
        if label > 0.5:
            for col, k in kernels.items():
                start = i - len(k) + 1
                if start >= 0:
                    df[col][start: i + 1] = k


def calc_output_bias(p_div_n):
    """
    Initial model bias for unbalanced data.

    :param p_div_n: positive rate divided by negative rate
    :type p_div_n: float
    :return:
    :rtype: float
    """
    if p_div_n is not None:
        bias = np.log([p_div_n])
        output_bias = tf.keras.initializers.Constant(bias)
    else:
        output_bias = None
    return output_bias


def conv_model(n_filters, conv_width, n_predictors, p_div_n=None):
    output_bias = calc_output_bias(p_div_n)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=n_filters, kernel_size=(conv_width,), activation='relu',
                               input_shape=[conv_width, n_predictors]),
        tf.keras.layers.Dense(units=n_filters, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)
    ])
    return model


def limited_filters_conv_model(n_filters, conv_width, n_predictors, p_div_n=None):
    """
    Want to limit the number of filters. Let:

        p = number of predictors = 100
        f = number of filters = 10
        k = kernel size = 3

    If we do Conv1D, each filter will have kxp weights and will produce one output. If we combine these outputs with a f
    to one dense layer, then the total number of weights is:

        k*p*f + f which is approximately p*(k*f)

        3*100*10 + 10 = 3010 weights


    If we stack the inputs into one vector, then the filter will have k elements. If we do a stride of k, then
    each output
    from Conv1D will come from one predictor. The number of outputs from all filters will be p*f. Combining these
    with one dense layer gives:

        k*f + f*p which is approximately p*f
        3x10 + 10*100 = 1030

    In other words, the first method has k times as many weights.
    """
    output_bias = calc_output_bias(p_div_n)
    model = tf.keras.Sequential([
        tf.keras.layers.Permute((2, 1)),
        tf.keras.layers.Reshape([1, conv_width * n_predictors, 1]),
        tf.keras.layers.Conv1D(filters=n_filters, kernel_size=(conv_width,), activation='relu', strides=(conv_width,)),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(units=5, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=output_bias)
    ])
    return model
