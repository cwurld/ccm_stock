# A runner to make imports work like they do in a Jupyter notebook

import matplotlib.pyplot as plt

from nn.nn import load_data, compile_and_fit_classifier
from nn.window_generators import IntermixedWindowGenerator
from nn.nn2 import get_results_dir, make_labels, CONV_WIDTH, diff_norm, limited_filters_conv_model
from my_utils.volatility import load_volatility


config = {
    'start_date': '20170103',
    'end_date': '20181231',
    'price_field': 'Open',
    'predictor_field': 'Open'
}
splits = [0.5, 0.75]
results_dir = get_results_dir(config)

# Load data
volatility = load_volatility(results_dir, config)
target = 'PLAG'
predictors = [x[0] for x in volatility[0:100]]
dataframe = load_data(target, predictors, config)
n_predictors = dataframe.shape[1]
n_filters = max(min(10, int(n_predictors / 4)), 4)

# label data
labels, threshold, frac_pos = make_labels(dataframe.target, CONV_WIDTH, frac_positive=0.09)
print('Percent positive: {}'.format(100 * frac_pos))
if labels is None:
    print('There are not enough price spikes in {}'.format(target))
else:
    # Normalize
    norm_df = diff_norm(dataframe)

    if 0:
        # Adds signal before each positive label for testing the model
        kernels = {
            predictors[0]: [0.05, -0.05, 1.0]
        }
        # For testing the model.
        add_signals(norm_df, labels, kernels)

    conv_window = IntermixedWindowGenerator(norm_df, labels, splits, CONV_WIDTH, balanced=True)

    # model = conv_model(n_filters, CONV_WIDTH, n_predictors)
    model = limited_filters_conv_model(n_filters, CONV_WIDTH, n_predictors)
    history = compile_and_fit_classifier(model, conv_window, patience=10, max_epochs=200, verbose=1)
    model.summary()

    print('n predictors: {}'.format(n_predictors))
    print('n filters: {}'.format(n_filters))
    print(predictors + [target])

    auc_train = model.evaluate(x=conv_window.train[0], y=conv_window.train[1], verbose=0)[7]
    auc_val = model.evaluate(x=conv_window.val[0], y=conv_window.val[1], verbose=0)[7]
    print('Train AUC: {}'.format(auc_train))
    print('Val   AUC: {}'.format(auc_val))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.title('Loss')
    plt.show()

    print('done')
