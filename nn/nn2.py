from pathlib import Path
from functools import partial

# Binary Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from nn import load_data, WindowGenerator, NormalizeAndSplit
from my_utils.volatility import load_volatility

RESULTS_DIR = Path(__file__).parent / 'results'


def get_results_dir(config):
    return RESULTS_DIR / '{}_{}'.format(config['start_date'], config['end_date'])


def multistep_dense(n_predictors, conv_width):
    n_units = n_predictors * conv_width
    model = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=n_units, activation='relu'),
        tf.keras.layers.Dense(units=n_units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def f():
    config = {
        'start_date': '20170103',
        'end_date': '20181231',
        'price_field': 'Open',
        'predictor_field': 'Open'
    }
    splits = [0.5, 0.75]
    results_dir = get_results_dir(config)

    target = 'MTL'
    # predictors = ['MTL', 'AGR', 'PHD', 'ESS', 'CBOE', 'AGRO', 'FPF', 'CHMA', 'SEAS']
    volatility = load_volatility(results_dir, config)
    predictors = [x[0] for x in volatility[0:20]]
    n_predictors = len(predictors)
    print(predictors)
    print('n predictors: {}'.format(n_predictors))
    dataframe = load_data(target, predictors, config)

    # Make classes: 1 if next price is greater than price, 0 otherwise
    diff = dataframe.target.diff(-1).to_numpy()
    labels = np.zeros(diff.shape[0])
    ii = np.where(diff < 0.0)
    labels[ii] = 1.0
    n = len(dataframe)
    split_labels = {
        'train': labels[0: int(n * splits[0])],
        'val': labels[int(n * splits[0]): int(n * splits[1])],
        'test': labels[int(n * splits[1]): None]
    }

    norm_and_split = NormalizeAndSplit(dataframe, splits, None)
    train_df, val_df, test_df = norm_and_split.no_norm()

    CONV_WIDTH = 3
    input_width = CONV_WIDTH
    label_width = 1
    shift = 1
    conv_window = WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df, 
                                  batch_size=1000, shuffle=False, labels=split_labels)

    # dataset = dataframe.values
    # # split into input (X) and output (Y) variables
    # X = dataset[:, 0:60].astype(float)
    # Y = dataset[:, 60]
    # # encode class values as integers
    # encoder = LabelEncoder()
    # encoder.fit(Y)
    # encoded_Y = encoder.transform(Y)

    # baseline model
    def create_baseline():
        # create model
        model = Sequential()
        model.add(Dense(60, input_dim=60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    x = conv_window.train
    build_fn = partial(multistep_dense, n_predictors, CONV_WIDTH)
    # evaluate baseline model with standardized dataset
    estimators = []
    # estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=build_fn, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, train_df.values, split_labels['train'], cv=kfold)
    print("Standardized: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))


if __name__ == '__main__':
    f()