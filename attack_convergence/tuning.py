import sys
sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from attack_convergence.core import ConvergenceAttack, ConvergenceWrapper
import parsing
import utils

def run_one(domain, model, attack_config_file, attack_name, parameter_overrides, save_path, stop):
    seed = 0 # 0 is for tuning, 1 is for actual tests
    track_every = 100
    num_samples = 250

    utils.set_seed(seed)

    # Prepare the attacks
    wrapped_model = ConvergenceWrapper(model, track_every, stop=stop)
    attack_config = utils.read_attack_config_file(attack_config_file)
    attack = parsing.parse_attack(attack_name, domain, float('inf'), 'standard', wrapped_model, attack_config, device='cuda', seed=seed, parameter_overrides=parameter_overrides)
    attack = ConvergenceAttack(wrapped_model, attack, float('inf'), False, suppress_warning=True)

    dataset = parsing.parse_dataset(domain, 'std:test', dataset_edges=(0, num_samples))
    dataloader = DataLoader(dataset, num_samples, num_workers=2, shuffle=False)

    individual_results = { index: { 'stats' : [] } for index in range(num_samples)}

    images, true_labels = next(iter(dataloader))
    assert len(images) == num_samples
    images = images.cuda()
    true_labels = true_labels.cuda()

    images, true_labels, labels = utils.apply_misclassification_policy(model, images, true_labels, 'use_predicted')
    stats, (final_step, final_found, final_distances) = attack(images, labels)

    for index in range(num_samples):
        individual_results[index]['final_stats'] = {
            'step' : final_step,
            'success' : final_found[index].item(),
            'distance' : final_distances[index].item() if final_found[index] else None
        }

        for step, found, distances in stats:
            individual_results[index]['stats'].append(
                (step, distances[index].item() if found[index].item() else None)
            )

    average_results = []

    last_step = max(x['final_stats']['step'] for x in individual_results.values())
    for step in [1] + list(range(track_every, last_step, track_every)) + [last_step]:
        best_success_at_step = []
        best_distances_at_step = []

        for index in range(num_samples):
            last_stat = [stat for stat in individual_results[index]['stats'] if stat[0] <= step][-1]
            if last_stat[1] is None:
                best_success_at_step.append(False)
            else:
                best_success_at_step.append(True)
                best_distances_at_step.append(last_stat[1])

        success_rate = np.count_nonzero(best_success_at_step) / len(best_success_at_step)
        average_distance = np.mean(best_distances_at_step)
        if np.isnan(average_distance):
            average_distance = None
        average_results.append(
            (step, success_rate, average_distance)
        )

    results = {
        # 'individual' : individual_results,
        'average' : average_results,
        'parameters' : parameter_overrides
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results, f)

def generate_overrides():
    all_overrides = {}

    standard_binary_search = []
    for max_eps in [0.1, 0.5]:
        for eps_initial_search_steps, eps_initial_search_factor in \
            [(20, 0.75), (10, 0.5), (5, 0.25), (0, 0)]:
            for eps_binary_search_steps in [5, 10, 15, 20]:
                standard_binary_search.append((max_eps, eps_initial_search_steps, eps_initial_search_factor, eps_binary_search_steps))
    standard_binary_search.append((1, 30, 0.75, 20))

    bim_overrides = []
    for nb_iter in [10, 50, 100, 200, 500]:
        for eps_iter in [1e-2, 1e-3, 1e-4, 1e-5]:
            for max_eps, eps_initial_search_steps, eps_initial_search_factor, eps_binary_search_steps in standard_binary_search:
                bim_overrides.append({
                    'nb_iter' : nb_iter,
                    'eps_iter': eps_iter,
                    'enable_binary_search' : True,
                    'max_eps' : max_eps,
                    'eps_initial_search_steps' : eps_initial_search_steps,
                    'eps_initial_search_factor' : eps_initial_search_factor,
                    'eps_binary_search_steps' : eps_binary_search_steps
                })

    # Extra sets
    for nb_iter in [10, 50, 100, 200, 500]:
        for eps_iter in [1e-1]:
            for max_eps, eps_initial_search_steps, eps_initial_search_factor, eps_binary_search_steps in standard_binary_search:
                bim_overrides.append({
                    'nb_iter' : nb_iter,
                    'eps_iter': eps_iter,
                    'enable_binary_search' : True,
                    'max_eps' : max_eps,
                    'eps_initial_search_steps' : eps_initial_search_steps,
                    'eps_initial_search_factor' : eps_initial_search_factor,
                    'eps_binary_search_steps' : eps_binary_search_steps
                })
    all_overrides['bim'] = bim_overrides

    brendel_overrides = []
    for steps in [10, 50, 100, 200, 500]:
        for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            for lr_num_decay in [25, 50, 100]:
                for momentum in [0, 0.5, 0.8]:
                    for binary_search_steps in [5, 10, 15]:
                        for initialization_attempts in [1, 5, 10]:
                            for init_directions in [1, 10, 100, 1000]:
                                for init_steps in [1, 10, 100, 1000]:
                                    brendel_overrides.append({
                                        'steps' : steps,
                                        'lr' : lr,
                                        'lr_num_decay' : lr_num_decay,
                                        'momentum' : momentum,
                                        'binary_search_steps' : binary_search_steps,
                                        'initialization_attempts' : initialization_attempts,
                                        'init_directions' : init_directions,
                                        'init_steps' : init_steps
                                    })
    all_overrides['brendel'] = brendel_overrides

    carlini_overrides = []
    for initial_tau in [0.5, 0.1]:
        for tau_factor in [0.9, 0.75, 0.5]:
            for max_const in [1e-4, 1e-2, 1, 100]:
                for learning_rate in [1e-2, 1e-3, 1e-4, 1e-5]:
                    for max_iterations in [10, 50, 100, 200, 500]:
                        carlini_overrides.append({
                            'initial_tau' : initial_tau,
                            'tau_factor' : tau_factor,
                            'max_const' : max_const,
                            'learning_rate' : learning_rate,
                            'max_iterations' : max_iterations,
                            'tau_check' : 1, # tau_check, const_check and inner_check massively slow down on CUDA, but are required for adequate evaluation
                            'const_check' : 1,
                            'inner_check' : 1,
                        })
    # Extra sets
    for initial_tau in [0.5, 0.1]:
        for tau_factor in [0.9, 0.75, 0.5]:
            for max_const in [1e-4, 1e-2, 1, 100]:
                for learning_rate in [1e-1]:
                    for max_iterations in [10, 50, 100, 200, 500]:
                        carlini_overrides.append({
                            'initial_tau' : initial_tau,
                            'tau_factor' : tau_factor,
                            'max_const' : max_const,
                            'learning_rate' : learning_rate,
                            'max_iterations' : max_iterations,
                            'tau_check' : 1, # tau_check, const_check and inner_check massively slow down on CUDA, but are required for adequate evaluation
                            'const_check' : 1,
                            'inner_check' : 1,
                        })
    all_overrides['carlini'] = carlini_overrides

    deepfool_overrides = []
    for steps in [100, 200, 500, 1000]:
        for overshoot in [1e-2, 1e-3, 1e-4, 1e-5]:
            deepfool_overrides.append({
                'steps' : steps,
                'overshoot' : overshoot
            })
    # Extra sets
    for steps in [100, 200, 500, 1000]:
        for overshoot in [1e-1]:
            deepfool_overrides.append({
                'steps' : steps,
                'overshoot' : overshoot
            })
    all_overrides['deepfool'] = deepfool_overrides

    fast_gradient_overrides = []
    for max_eps, eps_initial_search_steps, eps_initial_search_factor, eps_binary_search_steps in standard_binary_search:
        fast_gradient_overrides.append({
            'max_eps' : max_eps,
            'eps_initial_search_steps' : eps_initial_search_steps,
            'eps_initial_search_factor' : eps_initial_search_factor,
            'eps_binary_search_steps' : eps_binary_search_steps
        })
    all_overrides['fast_gradient'] = fast_gradient_overrides

    pgd_overrides = []
    for nb_iter in [10, 50, 100, 200, 500]:
        for eps_iter in [1e-3, 1e-4, 1e-5]:
            for max_eps, eps_initial_search_steps, eps_initial_search_factor, eps_binary_search_steps in standard_binary_search:
                pgd_overrides.append({
                    'nb_iter' : nb_iter,
                    'eps_iter' : eps_iter,
                    'max_eps' : max_eps,
                    'eps_initial_search_steps' : eps_initial_search_steps,
                    'eps_initial_search_factor' : eps_initial_search_factor,
                    'eps_binary_search_steps' : eps_binary_search_steps
                })
    # Extra sets
    for nb_iter in [10, 50, 100, 200, 500]:
        for eps_iter in [1e-1, 1e-2]:
            for max_eps, eps_initial_search_steps, eps_initial_search_factor, eps_binary_search_steps in standard_binary_search:
                pgd_overrides.append({
                    'nb_iter' : nb_iter,
                    'eps_iter' : eps_iter,
                    'max_eps' : max_eps,
                    'eps_initial_search_steps' : eps_initial_search_steps,
                    'eps_initial_search_factor' : eps_initial_search_factor,
                    'eps_binary_search_steps' : eps_binary_search_steps
                })
    all_overrides['pgd'] = pgd_overrides

    uniform_overrides = []
    for count in [10, 50, 100, 200, 500]:
        for max_eps, eps_initial_search_steps, eps_initial_search_factor, eps_binary_search_steps in standard_binary_search:
            uniform_overrides.append({
                'count' : count,
                'max_eps' : max_eps,
                'eps_initial_search_steps' : eps_initial_search_steps,
                'eps_initial_search_factor' : eps_initial_search_factor,
                'eps_binary_search_steps' : eps_binary_search_steps
            })
    all_overrides['uniform'] = uniform_overrides

    return all_overrides


@click.command()
@click.argument('domain', type=click.Choice(['mnist', 'cifar10']))
#@click.argument('attacks', callback=parsing.ParameterList(parsing.supported_attacks))
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='balanced_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
              'attack configuration.')
def main(domain, attack_config_file):
    utils.enable_determinism()
    stop = 10100
    architecture = 'a'
    test_type = 'standard'

    # Load the model
    if test_type == 'relu':
        state_dict_path = f'trained-models/classifiers/relu/relu-pruned/{domain}-{architecture}.pth'
    else:
        state_dict_path = f'trained-models/classifiers/{test_type}/{domain}-{architecture}.pth'

    model = parsing.parse_model(domain, architecture, state_dict_path, False, test_type == 'relu', False, True)
    model.eval()
    model.cuda()

    all_runs = []

    for attack_name, override_list in generate_overrides().items():
        for i, overrides in enumerate(override_list):
            all_runs.append((attack_name, i, overrides))

    pbar = tqdm(all_runs)
    for attack_name, i, overrides in pbar:
        pbar.set_description(attack_name)
        save_path = f'./attack_convergence/tuning_results/{domain}/{attack_name}/{i}.json'
        if not Path(save_path).exists():
            run_one(domain, model, attack_config_file, attack_name, overrides, save_path, stop)

if __name__ == '__main__':
    main()