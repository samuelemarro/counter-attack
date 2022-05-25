from email.policy import default
import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats

import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

import sklearn.metrics
import sklearn.linear_model
from sklearn.linear_model import QuantileRegressor

EIGHT_BIT_DISTANCE = 1/255


def get_distances(domain, architecture, test, parameter_set, atol, rtol):
    approximation_path = f'analysis/distances/{parameter_set}/{domain}-{architecture}-{test}.json'

    with open(approximation_path, 'r') as f:
        approximations = json.load(f)
    
    mip_path = f'analysis/distances/mip/{domain}-{architecture}-{test}.json'

    with open(mip_path, 'r') as f:
        mip_distances = json.load(f)

    assert set(mip_distances.keys()) == set(approximations.keys())

    target_distances = []
    approximation_distances = []

    for index, distances in approximations.items():
        if index not in mip_distances:
            raise RuntimeError
        upper = mip_distances[index]['upper']
        lower = mip_distances[index]['lower']

        if upper is None or lower is None and not (np.abs(upper - lower) <= atol or np.abs((upper - lower) / upper) <= rtol):
            continue

        valid_distances = [distance for distance in distances.values() if distance is not None]

        if len(valid_distances) == 0:
            raise RuntimeError

        target_distances.append(upper)
        approximation_distances.append(min(valid_distances))

    assert len(target_distances) == len(approximation_distances)

    return target_distances, approximation_distances

def get_calibration(train_approximation_distances, train_target_distances, val_approximation_distances, val_target_distances, quantile):
    qr = QuantileRegressor(quantile=quantile, alpha=0, solver='highs')
    qr.fit(train_approximation_distances.reshape(-1, 1), train_target_distances)
    predicted_target_distances = qr.predict(val_approximation_distances.reshape(-1, 1)).squeeze()

    num_leq = 0

    for predicted_target, val_target in zip(predicted_target_distances, val_target_distances):
        if val_target <= predicted_target:
            num_leq += 1
    
    return num_leq / len(val_target_distances), predicted_target_distances

def get_f1_scores(predicted_target_distances, actual_target_distances, all_eps):
    # True positive: predicted distance is > eps, actual distance is > eps
    # False positive: predicted distance is > eps, actual distance is <= eps
    # False negative: predicted distance is <= eps, actual distance is > eps
    # True negative: predicted distance is <= eps, actual distance is <= eps
    f1_values = []
    for eps in all_eps:
        ground_truth = np.less_equal(actual_target_distances, eps)
        predicted = np.less_equal(predicted_target_distances, eps)

        # print(np.count_nonzero(np.equal(ground_truth, predicted)) / len(ground_truth))

        if np.count_nonzero(ground_truth) == 0 and np.count_nonzero(predicted) == 0:
            f1_values.append(None)
        else:
            f1 = sklearn.metrics.f1_score(ground_truth, predicted)
            f1_values.append(f1)

    return f1_values

def calibration_cross_validation(target_distances, approximation_distances, quantile, k):
    target_distances = np.array(target_distances)
    approximation_distances = np.array(approximation_distances)

    indices = np.random.permutation(len(target_distances))

    folds = np.array_split(indices, k)

    all_actual_quantiles = []
    predicted_folds = []
    actual_folds = []

    for i in range(k):
        train_indices = np.concatenate([*folds[:i], *folds[i+1:]], axis=0)
        val_indices = folds[i]

        train_approximation_distances = approximation_distances[train_indices]
        train_target_distances = target_distances[train_indices]

        val_approximation_distances = approximation_distances[val_indices]
        val_target_distances = target_distances[val_indices]

        actual_quantile, predicted_target_distances = get_calibration(train_approximation_distances, train_target_distances, val_approximation_distances, val_target_distances, quantile)
        all_actual_quantiles.append(actual_quantile)
        actual_folds.append(val_target_distances)
        predicted_folds.append(predicted_target_distances)

    return np.mean(all_actual_quantiles), np.std(all_actual_quantiles), actual_folds, predicted_folds

def f1_plot(actual_folds, predicted_folds):
    min_distance = 0
    max_distance = max(max([max(actual_fold) for actual_fold in actual_folds]), max([max(predicted_fold) for predicted_fold in predicted_folds])) * 1.01
    print(min_distance)
    print(max_distance)
    all_eps = np.linspace(min_distance, max_distance, num=1000)

    all_f1_values = []

    for val_target_distances, predicted_target_distances in zip(actual_folds, predicted_folds):
        f1_values = get_f1_scores(predicted_target_distances, val_target_distances, all_eps)
        all_f1_values.append(f1_values)
    

    eps_for_plot = []
    f1s_for_plot = []

    for i in range(len(all_eps)):
        valid_f1s = [f1_values[i] for f1_values in all_f1_values if f1_values[i] is not None]
        if len(valid_f1s) > 0:
            eps_for_plot.append(all_eps[i])
            f1s_for_plot.append(np.mean(valid_f1s))

    return np.array(eps_for_plot), np.array(f1s_for_plot)

TEST_DISPLAY_NAMES = {
    'standard': 'Standard',
    'adversarial': 'Adversarial',
    'relu': 'ReLU'
}

@click.command()
@click.argument('domain')
@click.argument('parameter_set')
@click.argument('atol', type=float)
@click.argument('rtol', type=float)
@click.argument('cross_validation_folds', type=int)
@click.option('--test-override', type=str, default=None)
def main(domain, parameter_set, atol, rtol, cross_validation_folds, test_override):
    np.random.seed(0)
    #np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

    csv = 'Architecture,Training,Expected Quantile,True Quantile\n'

    for architecture in ['a', 'b', 'c']:
        for test in ['standard', 'adversarial', 'relu'] if test_override is None else [test_override]:
            print(f'{domain}-{architecture}-{test}')
            target_distances, approximation_distances = get_distances(domain, architecture, test, parameter_set, atol, rtol)

            for quantile in [0.01, 0.5, 1 - 0.01]:
                print(quantile * 100)
                mean, std, actual_folds, predicted_folds = calibration_cross_validation(target_distances, approximation_distances, quantile, cross_validation_folds)
                csv += f'{architecture.capitalize()},{TEST_DISPLAY_NAMES[test]},{quantile*100:.2f}\\%,{mean*100:.2f}\\textpm{std*100:.2f}\\%\n'

                # Save data for mean F1 plotting
                eps_for_plot, f1s_for_plot = f1_plot(actual_folds, predicted_folds)

                plot_csv = 'Epsilon,F1\n'
                for eps, f1 in zip(eps_for_plot, f1s_for_plot):
                    plot_csv += f'{eps},{f1}\n'
                
                plot_csv_path = f'analysis/calibration/plot/{domain}-{architecture}-{test}-{parameter_set}-{int(quantile * 100)}.csv'

                Path(plot_csv_path).parent.mkdir(parents=True, exist_ok=True)

                with open(plot_csv_path, 'w') as f:
                    f.write(plot_csv)
    print(csv)
    csv_path = f'analysis/calibration/prediction/{domain}-{parameter_set}.csv'
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write(csv)


if __name__ == '__main__':
    main()