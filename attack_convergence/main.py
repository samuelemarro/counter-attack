import sys
sys.path.append('.')

import json
from pathlib import Path

import click
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from attack_convergence.core import ConvergenceAttack, ConvergenceWrapper, CustomIndexedDataset
import parsing
import utils

def run_one(domain, model, attack_name, override_number, track_every, attack_config_file, parameter_overrides):
    # Prepare the attacks
    stop = 101000
    wrapped_model = ConvergenceWrapper(model, track_every, stop=stop)
    attack_config = utils.read_attack_config_file(attack_config_file)
    attack = parsing.parse_attack(attack_name, domain, float('inf'), 'standard', wrapped_model, attack_config, parameter_overrides=parameter_overrides, device='cuda')
    attack = ConvergenceAttack(wrapped_model, attack, float('inf'), False, suppress_warning=True)

    # Load the relevant indices
    with open(f'{domain}_indices_intersection.json') as f:
        relevant_indices = json.load(f)
    dataset = CustomIndexedDataset(parsing.parse_dataset(domain, 'std:test'), relevant_indices)
    dataloader = DataLoader(dataset, 250, num_workers=2, shuffle=False)

    results = { index: { 'stats' : [] } for index in relevant_indices}

    for indices, (images, true_labels) in tqdm(dataloader, desc=f'{attack_name}-{override_number}'):
        indices = indices.cuda()
        images = images.cuda()
        true_labels = true_labels.cuda()
        images, true_labels, labels = utils.apply_misclassification_policy(model, images, true_labels, 'use_predicted')
        assert len(images) == len(indices)
        stats, _ = attack(images, labels)
    
        for inner_index, actual_index in enumerate(indices):
            # inner_index: 0, 1, 2...
            # actual_index: 489, 491, 501...
            if isinstance(actual_index, torch.Tensor):
                actual_index = actual_index.item()

            for step, found, distances in stats:
                results[actual_index]['stats'].append(
                    (step, distances[inner_index].item() if found[inner_index].item() else None)
                )

    return results


@click.command()
@click.argument('domain', type=click.Choice(['mnist', 'cifar10']))
@click.argument('architecture', type=click.Choice(['a', 'b', 'c']))
@click.argument('test_type', type=click.Choice(['standard', 'adversarial', 'relu']))
@click.argument('track_every', type=click.IntRange(1, None))
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='balanced_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
              'attack configuration.')
def main(domain, architecture, test_type, track_every, attack_config_file):
    utils.enable_determinism()

    utils.set_seed(1) # 0 for tuning, 1 for actual tests

    # Load the model
    if test_type == 'relu':
        state_dict_path = f'trained-models/classifiers/relu/relu-pruned/{domain}-{architecture}.pth'
    else:
        state_dict_path = f'trained-models/classifiers/{test_type}/{domain}-{architecture}.pth'

    model = parsing.parse_model(domain, architecture, state_dict_path, False, test_type == 'relu', False, True)
    model.eval()
    model.cuda()

    with open(f'attack_convergence/best_overrides_{domain}.json') as f:
        overrides = json.load(f)

    for attack_name in ['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform']:
        for override_number in [100, 1000, 10000]:
            save_path = f'attack_convergence/attack_results/{domain}/{architecture}/{test_type}/{attack_name}/{override_number}.json'
            save_path = Path(save_path)

            if not save_path.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)

                override = overrides[attack_name][str(override_number)]
                results = run_one(domain, model, attack_name, override_number, track_every, attack_config_file, override)

                with open(str(save_path), 'w') as f:
                    json.dump(results, f)

if __name__ == '__main__':
    main()