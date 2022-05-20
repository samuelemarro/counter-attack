import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import torch

from adversarial_dataset import AdversarialDataset, AttackComparisonDataset, MergedComparisonDataset, MergedDataset
import utils


def get_indices(attack_dataset):
    if isinstance(attack_dataset, AttackComparisonDataset):
        if attack_dataset.indices_override is None:
            # Use start and stop
            return list(range(attack_dataset.start, attack_dataset.stop))
        else:
            # Use indices_override
            return list(attack_dataset.indices_override)
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
        # AdversarialDataset uses only one attack
        return { 'attack' : attack_dataset.adversarials[index - attack_dataset.start] }
    else:
        raise ValueError(f'Unknown attack dataset type: {type(attack_dataset)}')

@click.command()
@click.argument('dataset_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, file_okay=True, dir_okay=False))
def main(dataset_path, output_path):
    dataset = utils.load_zip(dataset_path)

    distances = {}

    if isinstance(dataset, AdversarialDataset):
        raise RuntimeError

    if isinstance(dataset, MergedComparisonDataset) or isinstance(dataset, AttackComparisonDataset):
        for index in get_indices(dataset):
            index_results = {}

            genuine = genuine_at_index(dataset, index)
            attack_results = attack_results_at_index(dataset, index)

            for attack_name, attack_result in attack_results.items():
                if attack_result is None:
                    index_results[attack_name] = None
                else:
                    index_results[attack_name] = torch.abs(attack_result - genuine).max().item()

            distances[index] = index_results
    elif isinstance(dataset, MergedDataset):
        for index in dataset.genuines.keys():
            index_results = {}

            genuine = dataset.genuines[index]
            lower_bound = dataset.lower_bounds[index]
            upper_bound = dataset.upper_bounds[index]

            distances[index] = {
                'lower' : lower_bound,
                'upper' : upper_bound
            }
    else:
        raise NotImplementedError

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(distances, f)

if __name__ == '__main__':
    main()