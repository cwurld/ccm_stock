# Based on: https://www.tensorflow.org/tutorials/structured_data/time_series
from abc import ABC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import leading_pattern.model as utils

MAX_EPOCHS = 200


class NormalizeAndSplit:
    def __init__(self, df, splits, target):
        self.df = df
        self.splits = splits
        self.target = target
        
        n = len(df)
        self.slices = {
            'train': slice(0, int(n * splits[0])),
            'val': slice(int(n * splits[0]), int(n * splits[1])),
            'test': slice(int(n * splits[1]), None)
        }

    def apply_splits(self, df):
        split_dfs = {}
        for df_name, the_slice in self.slices.items():
            split_dfs[df_name] = df[the_slice]
        return split_dfs

    def no_norm(self):
        ndf = self.df.copy()
        split_dfs = self.apply_splits(ndf)
        self.norm_method = 'no_norm'
        return tuple(split_dfs.values())

    def norm_by_mean(self):
        ndf = self.df.copy()
        split_dfs = self.apply_splits(ndf)

        self.train_mean = split_dfs['train'].mean()
        self.train_std = split_dfs['train'].std()

        for df_name in split_dfs:
            split_dfs[df_name] = (split_dfs[df_name] - self.train_mean) / self.train_std

        self.norm_method = 'norm_by_mean'
        return tuple(split_dfs.values())

    def moving_avg(self, window_width):
        """
        The first window_width-1 data points are removed from the data because there is not enough data
        for the rolling window.
        """
        self.window_width = window_width
        df = self.df.copy()
        self.moving_avg_dataframe = df.rolling(window_width).sum() / window_width
        normalized = (df - self.moving_avg_dataframe) / self.moving_avg_dataframe
        split_dfs = self.apply_splits(normalized)
        split_dfs['train'] = split_dfs['train'][window_width-1:]
        self.norm_method = 'moving_average'
        return tuple(split_dfs.values())

    def de_normalize(self, predictions, data_set_name=None, model_total_window_size=0):
        if self.norm_method == 'no_norm':
            pass
        elif self.norm_method == 'norm_by_mean':
            predictions = (predictions * self.train_std[self.target]) + self.train_mean[self.target]
        elif 'moving_average':
            if data_set_name:
                r = self.moving_avg_dataframe[self.target][self.slices[data_set_name]].to_numpy()
                if data_set_name == 'train':
                    r = r[self.window_width - 1:]  # remove the NAN
            else:
                r = self.moving_avg_dataframe[self.target].to_numpy()
                r = r[self.window_width - 1:]
            r = r[max(model_total_window_size - 1, 0):]
            predictions = predictions * r + r
        else:
            raise RuntimeError('Invalid norm type')
        return predictions


class WindowGenerator:
    """
    Based on: https://www.tensorflow.org/tutorials/structured_data/time_series
    """

    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)  # python builtin slice function
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]  # numpy array of indices

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]   # numpy array of indices

    def __repr__(self):
        return '\n'.join([
            'Total window size: {}'.format(self.total_window_size),
            'Input indices: {}'.format(self.input_indices),
            'Label indices: {}'.format(self.label_indices),
            'Label column name(s): {}'.format(self.label_columns)])

    def split_window(self, features):
        # features = [sample index, time, feature], time is 0 to total_window_size - 1
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]  # shift is implemented in labels_slice

        # Select the labels from all features
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        # https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape
        # Passing a None in the new shape allows any value for that axis:
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, shuffle=True, batch_size=32):
        data = np.array(data, dtype=np.float32)  # convert data frame into numpy 2D array (time, feature)

        # Create a dataset of sliding windows over a time series provided as array.
        # batch_size: Number of time series samples in each batch
        # shuffle: Whether to shuffle output samples or instead draw them in chronological order.
        # samples are selected randomly, without replacement, in batches until all elements in data
        # have been selected
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size)

        # ds is an iterator of batches. Split divides each batch into "inputs" and "labels"
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='target', max_subplots=3):
        # inputs - batch size, window size, feature
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel('{} [normed]'.format(plot_col))
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                # plot predictions for sample n
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def plot_fit(self, model, denorm, model_name):
        # set denorm=None to not denormalize the data
        fit_data = {}
        y_min = 10000.0
        y_max = 0.0
        for data_set in ['train', 'val', 'test']:
            the_data = self.make_dataset(getattr(self, data_set + '_df'), shuffle=False, batch_size=2000)
            fit_data[data_set] = {}
            for inputs, labels in the_data:
                if denorm:
                    predictions = denorm(model(inputs).numpy().flatten(), data_set, self.total_window_size)
                    expected = denorm(labels.numpy().flatten(), data_set, self.total_window_size)
                else:
                    predictions = model(inputs).numpy().flatten()
                    expected = labels.numpy().flatten()
                y_min = min([y_min, predictions.min(), expected.min()])
                y_max = max([y_max, predictions.max(), expected.max()])
                fit_data[data_set]['predictions'] = predictions
                fit_data[data_set]['expected'] = expected

        fig = plt.figure(figsize=(8, 8))
        fig.tight_layout(pad=20.0)
        plot_num = 1
        for data_set in ['train', 'val', 'test']:
            plt.subplot(3, 1, plot_num)
            plt.plot(fit_data[data_set]['predictions'], label='Predictions')
            plt.plot(fit_data[data_set]['expected'],  label='Actual')
            plt.title(model_name + ': ' + data_set)
            plt.legend()
            plt.ylim(y_min, y_max)
            plot_num += 1

        # print('x')


class Baseline(tf.keras.Model, ABC):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    # noinspection PyMethodOverriding
    def call(self, inputs):
        """
        The model's forward pass

        Returns the current value as the prediction
        """
        print('in call')
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def compile_and_fit(model, window, patience=2, max_epochs=MAX_EPOCHS, verbose=1):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    # Train on all the batches
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping], verbose=verbose)
    return history


def load_data(target, predictors, config):
    target_df = utils.get_data_set(target, config)

    if predictors:
        predictor_dfs = utils.get_data_sets(predictors, config)
        data = {symbol: values[config['predictor_field']] for symbol, values in predictor_dfs.items()}
    else:
        data = {}
    data['target'] = target_df[config['price_field']]
    df = pd.DataFrame(data)
    return df


def make_test_data(model_type, config):
    """Make data that matches models so we can see if the models are working as designed."""
    pf = config['predictor_field']

    if model_type == 'linear':
        # Linear model: target[t] = w1*target[t-1] + w2*predictor[t-1]
        #   predictor[t-1] = (target[t] - w1*target[t-1]) / w2
        #   predictor[t] = (target[t + 1] - w1*target[t]) / w2
        #   predictor[t] = target[t + 1]/w2 -(w1/w2) * target[t]
        target = utils.get_data_set('NPTN', config)  # this stock has interesting variation
        w1 = 0.5
        w2 = 0.2

        predictor = target.copy()
        predictor[pf] = (target[pf].shift(-1) / w2) - ((w1 / w2) * target[pf])
        df = pd.DataFrame({'target': target[pf], 'P1': predictor[pf]})
        weights = [np.array([[w1, w2]]).transpose(), np.array([0.0])]
    elif model_type == 'multistep_old':
        # The method below is not great because it can lead to wild oscillations in the predictor stock.
        # target[t]   = w1*target[t-1] + w2*target[t-2] + w3*target[t-3] + w4*predictor[t-1] +
        #               w5*predictor[t-2] + w6*predictor[t-3]
        # target[t+1] = w1*target[t] +   w2*target[t-1] + w3*target[t-2] + w4*predictor[t] +
        #               w5*predictor[t-1] + w6*predictor[t-2]
        # w4*predictor[t] = target[t+1] - w1*target[t] -   w2*target[t-1] - w3*target[t-2] -
        #                   w5*predictor[t-1] - w6*predictor[t-2]
        # predictor[t] = (target[t+1] - w1*target[t] -   w2*target[t-1] - w3*target[t-2] -
        #                w5*predictor[t-1] - w6*predictor[t-2]) / w4
        target = utils.get_data_set('NPTN', config)  # this stock has interesting variation
        w = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
        n = len(target[pf])
        p = np.zeros(n)
        p[0] = 10.0
        p[1] = 10.0
        p[-1] = 0.0
        for t in range(2, n - 1):
            p[t] = (target[pf][t+1] - w[0]*target[pf][t] - w[1]*target[pf][t-1] - w[2]*target[pf][t-2] -
                    w[4]*p[t-1] - w[5]*p[t-2]) / w[3]
            # x = w[0]*target[pf][t] + w[1]*target[pf][t-1] + w[2]*target[pf][t-2] + \
            #     w[3]*p[t] + w[4]*p[t-1] + w[5]*p[t-2]
            # print(t, x, target[pf][t+1])
            # print('x')

        predictor = target.copy()
        predictor[pf] = p
        df = pd.DataFrame({'target': target[pf][3:], 'P1': predictor[pf][3:]})
        weights = [np.array([[0.2, 0.1, 0.2, 0.1, 0.2, 0.1]]).transpose(), np.array([0.0])]
    elif model_type == 'multistep':
        # In this approach, we will synthesize the target from the predictor. Hopefully that will be more stable.
        predictor = utils.get_data_set('NPTN', config)  # this stock has interesting variation
        p = predictor[pf]  # make the target we loaded into the predictor

        w = np.array([0.4, 0.3, 0.2, 0.3, -0.1, 2.0])
        n = len(p)
        target = np.zeros(n)
        target[0] = 10.0
        t = 1
        target[1] = w[0] * target[t - 1] + w[3] * p[t - 1]
        t = 2
        target[2] = w[0] * target[t - 1] + w[1] * target[t - 2] + w[3] * p[t - 1] + w[4] * p[t - 2]

        for t in range(3, n):
            target[t] = w[0] * target[t - 1] + w[1] * target[t - 2] + w[2] * target[t - 3] + \
                        w[3] * p[t - 1] + w[4] * p[t - 2] + w[5] * p[t - 3]

        if 0:
            plt.plot(target[3:])
            plt.show()
        df = pd.DataFrame({'target': target[3:], 'P1': predictor[pf][3:]})
        weights = [np.array([[w[2], w[5], w[1], w[4], w[0], w[3]]]).transpose(), np.array([0.0])]
    else:
        raise RuntimeError('Invalid model type')
    return df, weights


def skunk():
    # skunk) keeps variables out of global name-space
    config = {
        'start_date': '20170103',
        'end_date': '20181231',
        'price_field': 'Open',
        'predictor_field': 'Open'
    }
    splits = [0.5, 0.75]

    data_gen = 'real'
    plot_stocks = 0
    run_baseline = 1
    plot_baseline = 0
    run_linear_single_step = 1
    apply_weights = 0
    run_linear_multi_step = 0
    run_multi_step_dense = 0
    run_conv_model = 0
    # noinspection PyPep8Naming
    run_LSTM_model = 0

    if data_gen == 'real':
        target = 'NPTN'
        predictors = ['MTL', 'AGR', 'PHD', 'ESS', 'CBOE', 'AGRO', 'FPF', 'CHMA', 'SEAS']
        n_predictors = len(predictors) + 1
        df = load_data(target, predictors, config)
        norm_and_split = NormalizeAndSplit(df, splits, 'target')
        train_df, val_df, test_df = norm_and_split.moving_avg(10)
        weights = None
    else:
        n_predictors = 2
        df, weights = make_test_data(data_gen, config)
        norm_and_split = NormalizeAndSplit(df, splits, 'target')
        train_df, val_df, test_df = norm_and_split.moving_avg(10)

    column_indices = {name: i for i, name in enumerate(df.columns)}

    if plot_stocks:  # Plot the stocks ----------------------------------------------------------------------------
        plot_cols = ['target', 'MTL', 'PHD']
        plot_features = df[plot_cols]  # a dataframe subset
        plot_features.plot(subplots=True)
        plt.show()

        # plot all - visually, target is not a linear combination of predictors
        df.plot(subplots=True)
        plt.show()

    # ------------------------------------------------------------------------------------------------------------
    # Make a window that has one time as an input, one time as an output with a shift of one between them
    input_width = 1
    label_width = 1
    shift = 1
    single_step_window = \
        WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df, label_columns=['target'])

    input_width = 24
    label_width = 24  # one label for each input, shifted by 1
    shift = 1
    wide_window = WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df, label_columns=['target'])

    # Models ----------------------------------------------------------------------------------------------------
    mean_absolute_error = []

    # baseline model
    if run_baseline:
        baseline = Baseline(label_index=column_indices['target'])
        baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])

        # We do not need to "train" this model because it simply returns the current value as the prediction
        perf = [
            'Baseline',
            baseline.evaluate(single_step_window.train, verbose=0)[1],
            baseline.evaluate(single_step_window.val, verbose=0)[1],
            baseline.evaluate(single_step_window.test, verbose=0)[1]
        ]
        mean_absolute_error.append(perf)

        if plot_baseline:
            single_step_window.plot_fit(baseline, norm_and_split.de_normalize, 'Baseline')
            plt.show()

    # Linear - apply to each sample in single step window.
    if run_linear_single_step:
        # A dense layer has an output shape of (batch_size,units)
        linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

        if data_gen == 'linear':
            compile_and_fit(linear, single_step_window, max_epochs=1)
            linear.layers[0].set_weights(weights)
        else:
            compile_and_fit(linear, single_step_window, max_epochs=MAX_EPOCHS)

        perf = [
            'Linear',
            linear.evaluate(single_step_window.train, verbose=0)[1],
            linear.evaluate(single_step_window.val, verbose=0)[1],
            linear.evaluate(single_step_window.test, verbose=0)[1]
        ]
        mean_absolute_error.append(perf)
        
        single_step_window.plot_fit(linear, norm_and_split.de_normalize, 'Linear')
        plt.show()

        # Plot weights
        if 1:
            plt.bar(x=range(len(train_df.columns)), height=linear.layers[0].kernel[:, 0].numpy())
            axis = plt.gca()
            axis.set_xticks(range(len(train_df.columns)))
            axis.set_xticklabels(train_df.columns, rotation=90)
            plt.show()

    # Make a window for multi-step models
    # noinspection PyPep8Naming
    CONV_WIDTH = 3
    input_width = CONV_WIDTH
    label_width = 1
    shift = 1
    conv_window = WindowGenerator(input_width, label_width, shift, train_df, val_df, test_df, label_columns=['target'])

    if apply_weights:
        # A sanity check where we apply the weights directly to the data from conv_window.
        w = weights[0].flatten()
        for inputs, labels in conv_window.train:
            for i, the_input in enumerate(inputs):
                i2 = the_input.numpy().flatten()
                p = np.sum(i2 * w)
                print('Predicted: {}  Actual: {}'.format(p, labels[i].numpy()[0][0]))

    print('Conv window:')
    print(conv_window)

    if run_linear_multi_step:
        # This allows us to manually set the weights to confirm everything is working as expected.
        linear_multi_step = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])

        if data_gen == 'multistep':
            compile_and_fit(linear_multi_step, conv_window, max_epochs=300)
            linear_multi_step.layers[1].set_weights(weights)
        else:
            compile_and_fit(linear_multi_step, conv_window, max_epochs=MAX_EPOCHS)

        perf = [
            'Multi step linear',
            linear_multi_step.evaluate(conv_window.train, verbose=0)[1],
            linear_multi_step.evaluate(conv_window.val, verbose=0)[1],
            linear_multi_step.evaluate(conv_window.test, verbose=0)[1]
        ]
        mean_absolute_error.append(perf)

        conv_window.plot_fit(linear_multi_step, None, 'Multistep Linear')
        plt.show()

    # Multistep dense
    if run_multi_step_dense:
        n_units = n_predictors * CONV_WIDTH
        multi_step_dense = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=n_units, activation='relu'),
            tf.keras.layers.Dense(units=n_units, activation='relu'),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])

        compile_and_fit(multi_step_dense, conv_window, patience=10, max_epochs=500)

        perf = [
            'Multi step dense',
            multi_step_dense.evaluate(conv_window.train, verbose=0)[1],
            multi_step_dense.evaluate(conv_window.val, verbose=0)[1],
            multi_step_dense.evaluate(conv_window.test, verbose=0)[1]
        ]
        mean_absolute_error.append(perf)
        conv_window.plot_fit(multi_step_dense, None, 'Multistep Dense')
        plt.show()

    # Convolution model
    if run_conv_model:
        n_filters = n_predictors * 2
        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=n_filters, kernel_size=(CONV_WIDTH,), activation='relu'),
            tf.keras.layers.Dense(units=n_filters, activation='relu'),
            tf.keras.layers.Dense(units=1),
        ])
        compile_and_fit(conv_model, conv_window, patience=100, max_epochs=300)

        perf = [
            'Conv',
            conv_model.evaluate(conv_window.train, verbose=0)[1],
            conv_model.evaluate(conv_window.val, verbose=0)[1],
            conv_model.evaluate(conv_window.test, verbose=0)[1]
        ]
        mean_absolute_error.append(perf)
        
        conv_window.plot_fit(conv_model, norm_and_split.de_normalize, 'Conv')
        plt.show()

    # LSTM model
    if run_LSTM_model:
        lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])

        window = wide_window
        compile_and_fit(lstm_model, window, patience=10)

        perf = [
            'LSTM',
            lstm_model.evaluate(window.train, verbose=0)[1],
            lstm_model.evaluate(window.val, verbose=0)[1],
            lstm_model.evaluate(window.test, verbose=0)[1]
        ]
        mean_absolute_error.append(perf)

        # window.plot_fit(lstm_model, norm_and_split.de_normalize, 'LSTM')
        window.plot(lstm_model)
        plt.show()

    p2_df = pd.DataFrame(mean_absolute_error, columns=['Model', 'train', 'val', 'test'])
    print(p2_df)


if __name__ == '__main__':
    skunk()