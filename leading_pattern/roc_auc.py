import pickle
import json
import datetime
import time
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import leading_pattern.model as model
from utils import big_results_writer
import leading_pattern.plotting as plotting


def calc_fpr_tpr(target_df, predictor_df, config):
    """
    If at time t, r^2 is greater than or equal to threshold and (price[t + spike length + 1) - price[t])/price[t]
    is greater than the price change threshold, then this is a true positive.
    """
    prices = target_df[config['price_field']].to_numpy()
    price_threshold = config['spike_threshold_percent'] / 100.0

    thresholds = np.arange(0.95, 0.1, -0.1)
    tpr = np.empty(thresholds.shape[0])
    fpr = np.empty(thresholds.shape[0])
    ppv = np.empty_like(thresholds)
    npv = np.empty_like(thresholds)

    for j, threshold in enumerate(thresholds):
        real_positives = 0.0
        real_negatives = 0.0
        true_positives = 0.0
        false_negatives = 0.0
        false_positives = 0.0
        true_negatives = 0.0
        for i, r_squared in enumerate(predictor_df.r_squared):
            if config['predictor_func_length'] <= i < prices.shape[0] - (config['spike_length'] + 1):
                fraction_change = (prices[i + config['spike_length'] + 1] - prices[i]) / prices[i]
                if fraction_change >= price_threshold:
                    real_positives += 1.0
                    if r_squared >= threshold:
                        true_positives += 1.0
                    else:
                        false_negatives += 1.0
                else:
                    real_negatives += 1
                    if r_squared >= threshold:
                        false_positives += 1.0
                    else:
                        true_negatives += 1.0

        tpr[j] = true_positives / real_positives
        fpr[j] = false_positives / real_negatives

        d = (true_positives + false_positives)
        if d > 0:
            ppv[j] = true_positives / d
        else:
            ppv[j] = 0.0

        d = (true_negatives + false_negatives)
        if d > 0:
            npv[j] = true_negatives / d
        else:
            npv[j] = 0.0

    # Integrate to get AUC
    tpr_2 = np.insert(tpr, 0, 0.0)
    tpr_2 = np.append(tpr_2, [1.0])
    fpr_2 = np.insert(fpr, 0, 0.0)
    fpr_2 = np.append(fpr_2, [1.0])
    auc = np.trapz(tpr_2, fpr_2)

    df = pd.DataFrame({
        'threshold': thresholds,
        'fpr': fpr,
        'tpr': tpr,
        'ppv': ppv,
        'npv': npv
    })
    df.attrs['auc'] = auc
    return df


def calc_fpr_tpr_all(target_stock, config, dry_run=False):
    target_df = model.get_data_set(target_stock, config)
    model.find_spikes(target_df['train'], config)

    def func_by_item(predictor_stock):
        predictor_df = model.get_data_set(predictor_stock, config)
        predictor_func = model.get_predictor_func(target_df['train'], predictor_df['train'], config)
        results = []
        for name in ['train', 'test']:
            model.calc_r_squared(predictor_df[name], predictor_func, config)
            auc_df = calc_fpr_tpr(target_df[name], predictor_df[name], config)
            results.append(auc_df.attrs['auc'])
        return results

    out_path = model.get_results_dir(config) / 'auc_{}_{}.pickle'.format(target_stock, config['price_field'])
    predictor_stocks = model.get_all_symbols_in_cache(config)
    big_results_writer(out_path, predictor_stocks, func_by_item, dry_run=dry_run)


def summarize_auc(target_stock, config, max_rows=100):
    results_dir = model.get_results_dir(config)
    pickle_path = results_dir / 'auc_{}_{}.pickle'.format(target_stock, config['price_field'])
    with open(pickle_path, 'rb') as fp:
        auc_by_stock = pickle.load(fp)

    auc_list = sorted([[k] + v for k, v in auc_by_stock.items()], key=lambda x: x[2], reverse=True)
    auc_list.sort(key=lambda x: x[1], reverse=True)
    best_items = []
    n_rows = 0
    for item in auc_list:
        if item[1] < 0.6:
            break

        if item[2] > 0.5:
            best_items.append(item)
            n_rows += 1

        if n_rows >= max_rows:
            break

    with open(results_dir / 'auc_sum_{}_{}.json'.format(target_stock, config['price_field']), 'w') as fp:
        json.dump(best_items, fp, indent=4)

    df = pd.DataFrame(best_items, columns=['predictor', 'train', 'test'])
    return df


def run_all_predictors_for_target(target, config, dry_run=False):
    calc_fpr_tpr_all(target, config, dry_run=dry_run)
    summarize_auc(target, config)


def run_all_predictors_for_all_targets(config):
    results_dir = model.get_results_dir(config)
    with open(results_dir / 'spikes.json', 'r') as fp:
        spikes_list = json.load(fp)

    for target, n_training_spikes, n_test_spikes in spikes_list:
        full_path = results_dir / 'auc_{}_{}.pickle'.format(target, config['price_field'])
        if not os.path.exists(full_path) and n_training_spikes > 10 and n_test_spikes > 10:
            start = time.time()
            print('Target: {}  {}'.format(target, datetime.datetime.now().time()))
            run_all_predictors_for_target(target, config)
            print('Elapsed time: {:.2f} seconds'.format(time.time() - start))


def get_best_targets(config):
    results_dir = model.get_results_dir(config)

    for path in glob.glob(str(results_dir / 'auc_sum*')):
        with open(path, 'r') as fp:
            results_list = json.load(fp)

        for symbol, train_score, test_score in results_list:
            if train_score > 0.65 and test_score > 0.65:
                target = path.split('_')[-2]
                print(target, symbol, train_score, test_score)


def skunk():
    # Keeps variables out of global name-space
    np.seterr(all='raise')  # make code stop on warnings

    config = {
        'start_date': '20170103',
        'split_date': '20171231',
        'end_date': '20181231',
        'spike_length': 2,
        'spike_threshold_percent': 3.0, 
        'predictor_func_length': 7,
        'use_diffs': True,  # when True the predictor functions are a times series of diffs
        'price_field': 'Open',
        'predictor_field': 'Open'
    }

    if 0:
        # Run one predictor
        target = 'SEAS'
        predictor = 'HRZN'
        data_sets = model.get_data_sets([target] + [predictor], config)
        predictor_func_by_stock = model.get_predictor_func_by_stock(data_sets['train'], target, [predictor], config)
        for ns in ['train', 'test']:
            model.calc_r_squared_for_list(data_sets[ns], predictor_func_by_stock, config)
            auc_df = calc_fpr_tpr(data_sets[ns][target], data_sets[ns][predictor], config)
            plotting.plot_roc(auc_df, predictor, data_set_name=ns)
            plt.show()
            print('Data for {}  AUC: {:.2f}'.format(ns, auc_df.attrs['auc']))
            print(auc_df)

    if 0:
        # Run multiple predictors
        data_set_name = 'test'
        target = 'NPTN'
        predictors = ['NPTN', 'MTL', 'AGR', 'PHD', 'ESS', 'CBOE', 'AGRO', 'FPF', 'CHMA', 'SEAS']
        data_sets = model.get_data_sets([target] + predictors, config)
        predictor_func_by_stock = model.get_predictor_func_by_stock(data_sets['train'], target, predictors, config)
        auc_df_by_predictor = {}
        for predictor, p_func in predictor_func_by_stock.items():
            model.calc_r_squared_for_list(data_sets[data_set_name], predictor_func_by_stock, config)
            auc_df_by_predictor[predictor] = \
                calc_fpr_tpr(data_sets[data_set_name][target], data_sets[data_set_name][predictor], config)
        plotting.plot_roc_panel(auc_df_by_predictor, target, config)
        plt.show()

    if 1:
        # Run all predictors against target
        target = 'PLSE'
        start = time.time()
        print('Target: {}  {}'.format(target, datetime.datetime.now().time()))
        run_all_predictors_for_target(target, config, dry_run=False)
        print('{} Elapsed time: {:.2f} seconds'.format(target, time.time() - start))

    if 0:
        # Run all predictors against all targets
        run_all_predictors_for_all_targets(config)

    if 0:
        target = 'NPTN'
        df = summarize_auc(target, config, max_rows=10)
        print(df)

    if 0:
        get_best_targets(config)

    return None


if __name__ == '__main__':
    # noinspection PyNoneFunctionAssignment
    r = skunk()
    print('done')
