import sys

sys.path.append('.')

import click
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.stats
import torch

from adversarial_dataset import AdversarialDataset, AttackComparisonDataset, MergedComparisonDataset
import utils

def get_indices(attack_dataset):
    if isinstance(attack_dataset, AttackComparisonDataset):
        if attack_dataset.indices_override is None:
            # Use start and stop
            return list(range(attack_dataset.start, attack_dataset.stop))
        else:
            # Use indices_override
            return attack_dataset.indices_override
    elif isinstance(attack_dataset, MergedComparisonDataset):
        # Use dictionary keys
        return list(attack_dataset.genuines.keys())
    elif isinstance(attack_dataset, AdversarialDataset):
        # Use start and stop
        return list(range(attack_dataset.start, attack_dataset.stop))
    else:
        raise ValueError(f'Unknown attack dataset type: {type(attack_dataset)}')

def genuine_at_index(attack_dataset, index):
    if isinstance(attack_dataset, AttackComparisonDataset):
        if attack_dataset.indices_override is None:
            # Use start and stop
            return attack_dataset.genuines[index - attack_dataset.start]
        else:
            # Use indices_override
            return attack_dataset.genuines[attack_dataset.indices_override.index(index)]
    elif isinstance(attack_dataset, MergedComparisonDataset):
        return attack_dataset.genuines[index]
    elif isinstance(attack_dataset, AdversarialDataset):
        # Use start and stop
        return attack_dataset.genuines[index - attack_dataset.start]
    else:
        raise ValueError(f'Unknown attack dataset type: {type(attack_dataset)}')

def attack_results_at_index(attack_dataset, index):
    if isinstance(attack_dataset, AttackComparisonDataset):
        if attack_dataset.indices_override is None:
            # Use start and stop
            return attack_dataset.attack_results[index - attack_dataset.start]
        else:
            # Use indices_override
            return attack_dataset.attack_results[attack_dataset.indices_override.index(index)]
    elif isinstance(attack_dataset, MergedComparisonDataset):
        return attack_dataset.attack_results[index]
    elif isinstance(attack_dataset, AdversarialDataset):
        return { 'attack' : attack_dataset.adversarials[index - attack_dataset.start] }
    else:
        raise ValueError(f'Unknown attack dataset type: {type(attack_dataset)}')

@click.command()
@click.argument('name', type=str)
@click.argument('mip_dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('compare_dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('threshold', type=float)
def main(name, mip_dataset_path, compare_dataset_path, threshold):
    mip_dataset = utils.load_zip(mip_dataset_path)
    compare_dataset = utils.load_zip(compare_dataset_path)

    stats = []

    if isinstance(compare_dataset, AdversarialDataset):
        attack_names = ['attack']
    else:
        attack_names = ['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform']

    for attack_subset in utils.powerset(attack_names, False):
        mip_upper_bounds = []
        compare_distances = []

        for index in set(mip_dataset.genuines.keys()).intersection(get_indices(compare_dataset)):
            upper_bound = mip_dataset.upper_bounds[index]
            lower_bound = mip_dataset.lower_bounds[index]

            mip_genuine = mip_dataset.genuines[index]
            compare_genuine = genuine_at_index(compare_dataset, index)

            assert torch.abs(mip_genuine - compare_genuine).max() < 1e-5

            if upper_bound is None or lower_bound is None or upper_bound > 1e40 or (upper_bound - lower_bound > threshold):
                continue
            
            attack_results = attack_results_at_index(compare_dataset, index)

            successful_adversarials = [attack_results[attack_name] for attack_name in attack_subset if attack_results[attack_name] is not None]
            
            if len(successful_adversarials) == 0:
                continue

            successful_adversarial_distances = utils.one_many_adversarial_distance(compare_genuine, torch.stack(successful_adversarials), np.inf)
            best_successful_adversarial_distance = torch.min(successful_adversarial_distances)

            mip_upper_bounds.append(upper_bound)
            compare_distances.append(best_successful_adversarial_distance)

        print('Attack subset:', attack_subset)

        plt.clf()
        
        mip_upper_bounds = np.array(mip_upper_bounds)
        compare_distances = np.array(compare_distances)

        plt.scatter(mip_upper_bounds, compare_distances)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mip_upper_bounds, compare_distances)

        print('Slope:', slope)
        print('Intercept:', intercept)
        print('R-squared:', r_value ** 2)
        print('P-value:', p_value)
        print('Std-err:', std_err)

        x = np.linspace(0, np.max(mip_upper_bounds) * 1.05, 100)
        y = x * slope + intercept
        plt.plot(x, y, '--r')
        plt.plot(x, x, '--k')

        plt.xlabel('MIP upper bound')
        plt.ylabel('Best adversarial distance')
        plt.title(', '.join(attack_subset))

        print('=' * 50)

        stats.append({
            'attack_subset' : attack_subset,
            'slope' : slope,
            'intercept' : intercept,
            'r_value' : r_value,
            'r_squared' : r_value ** 2,
            'p_value' : p_value,
            'std_err' : std_err
        })

        figure_path = Path('analysis/results/approximation') / name / ('_'.join(attack_subset) + '.png')

        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path)

    stats_path = Path('analysis/results/approximation') / name / 'stats.json'

    with open(str(stats_path), 'w') as f:
        json.dump(stats, f)

if __name__ == '__main__':
    main()