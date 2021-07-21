from pathlib import Path
import random

import tensorflow as tf

RESULTS_DIR = Path(__file__).parent / 'results'


def get_results_dir(config):
    return RESULTS_DIR / '{}_{}'.format(config['start_date'], config['end_date'])


class IntermixedWindowGenerator:
    def __init__(self, df, labels, splits, window_width, min_run_length=10, balanced=False, seed=1):
        self.balanced = balanced
        self.seed = seed

        data = df.to_numpy()
        self.inputs = {
            'train': [],
            'validation': [],
            'test': []
        }

        self.outputs = {
            'train': [],
            'validation': [],
            'test': []
        }

        targets = {
            'train': int(min_run_length * (splits[0] / (splits[1] - splits[0]))),
            'validation': min_run_length,
            'test': 10000000
        }

        target = 'train'
        run_length = targets[target]
        test_threshold = int(len(df) * splits[-1])

        i = window_width
        run_length_count = 0
        while i < len(df):
            self.inputs[target].append(data[i - window_width: i])
            self.outputs[target].append(labels[i - 1])

            if target != 'test' and i > test_threshold:
                target = 'test'
                run_length_count = 0
                run_length = targets[target]
                i += window_width
            elif run_length_count == run_length:
                run_length_count = 0
                if target == 'train':
                    target = 'validation'
                else:
                    target = 'train'
                run_length = targets[target]
                i += window_width
            else:
                i += 1
                run_length_count += 1

        self.n_positive = {}
        self.n_negative = {}
        for data_set_name in ['train', 'validation', 'test']:
            self.n_positive[data_set_name] = len([v for v in self.outputs[data_set_name] if v > 0.5])
            self.n_negative[data_set_name] = len(self.outputs[data_set_name]) - self.n_positive[data_set_name]

        if self.balanced:
            self.balance_data()

    def balance_data(self):
        random.seed(self.seed)
        self.balanced_inputs = {
            'train': [],
            'validation': []
        }

        self.balanced_outputs = {
            'train': [],
            'validation': []
        }

        for key in self.balanced_outputs:
            if self.n_negative[key] > self.n_positive[key]:
                prob_neg = self.n_positive[key] / self.n_negative[key]
                prob_pos = 1.01
            else:
                prob_neg = 1.0
                prob_pos = self.n_negative[key] / self.n_positive[key]

            for the_input, the_output in zip(self.inputs[key], self.outputs[key]):
                x = random.random()
                if (the_output > 0.5 and x < prob_pos) or (the_output < 0.5 and x < prob_neg):
                    self.balanced_inputs[key].append(the_input)
                    self.balanced_outputs[key].append(the_output)

    def for_fitting(self, data_set_name):
        if self.balanced:
            n = len(self.balanced_outputs[data_set_name])
            outputs = tf.reshape(tf.convert_to_tensor(self.balanced_outputs[data_set_name]), (n, 1, 1))
            inputs = tf.convert_to_tensor(self.balanced_inputs[data_set_name])
        else:
            n = len(self.outputs[data_set_name])
            outputs = tf.reshape(tf.convert_to_tensor(self.outputs[data_set_name]), (n, 1, 1))
            inputs = tf.convert_to_tensor(self.inputs[data_set_name])
        return inputs, outputs

    @property
    def train(self):
        return self.for_fitting('train')

    @property
    def val(self):
        return self.for_fitting('validation')

    @property
    def test(self):
        return self.for_fitting('test')

    @property
    def pos_div_neg(self):
        if self.balanced:
            return 1.0
        else:
            return self.n_positive['train'] / self.n_negative['train']
