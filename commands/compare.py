import logging

import click
import numpy as np
import torch

import parsing
import tests
import utils

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('attacks', callback=parsing.ParameterList(parsing.supported_attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='default_attack_configuration.cfg', show_default=True)
@click.option('--keep-misclassified', is_flag=True)
@click.option('--max-samples', type=click.IntRange(1, None), default=None)
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True)
def compare(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], max_samples=kwargs['max_samples'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    p = kwargs['p']

    attack_names = kwargs['attacks']
    attacks = []

    for attack_name in attack_names:
        attack = parsing.get_attack(attack_name, kwargs['domain'], p, 'standard', model, attack_config)
        attacks.append(attack)

    result_dataset = tests.multiple_attack_test(model, attack_names, attacks, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs)
    
    #adversarial_dataset = tests.attack_test(model, attack_pool, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, None)

    print('===Standard Result===')
    complete_pool = result_dataset.simulate_pooling(attack_names)
    complete_pool.print_stats()
    print()

    # How much does a single attack contribute to the overall quality?
    print('===Attack Dropping Effects===')

    for attack_name in attack_names:
        other_attack_names = [x for x in attack_names if x != attack_name]
        pool_adversarial_dataset = result_dataset.simulate_pooling(other_attack_names)

        print('Without {}:'.format(attack_name))

        pool_adversarial_dataset.print_stats()
        print()

    attack_powerset = utils.powerset(attack_names)

    print('===Pool Stats===')
    for attack_set in attack_powerset:
        print('Pool {}:'.format(attack_set))

        pool_adversarial_dataset = result_dataset.simulate_pooling(attack_set)
        pool_adversarial_dataset.print_stats()
        print()

    print()
    print('===Best Pools===')
    print()

    for n in range(1, len(attack_names) + 1):
        print('==Pool of size {}=='.format(n))
        print()

        n_size_sets = [subset for subset in attack_powerset if len(subset) == n]
        n_size_pools = [result_dataset.simulate_pooling(subset) for subset in n_size_sets]

        attack_success_rates = np.array([x.attack_success_rate for x in n_size_pools])
        median_distances = np.array([np.median(x.distances) for x in n_size_pools])
        average_distances = np.array([np.average(x.distances) for x in n_size_pools])

        best_by_success_rate = np.argmax(attack_success_rates)

        print('Best pool of size {} by success rate: {}'.format(n, n_size_sets[best_by_success_rate]))
        n_size_pools[best_by_success_rate].print_stats()
        print()

        best_by_median_distance = np.argmin(median_distances)

        print('Best pool of size {} by median distance: {}'.format(n, n_size_sets[best_by_median_distance]))
        n_size_pools[best_by_median_distance].print_stats()
        print()

        best_by_average_distance = np.argmin(average_distances)
        print('Best pool of size {} by average distance: {}'.format(n, n_size_sets[best_by_average_distance]))
        n_size_pools[best_by_average_distance].print_stats()
        print()

    print('===Attack Ranking Stats===')

    for attack_name in attack_names:
        print('Attack {}:'.format(attack_name))

        attack_ranking_stats = result_dataset.attack_ranking_stats(attack_name)

        for position, rate in [x for x in attack_ranking_stats.items() if x[0] != 'failure']:
            print('The attack is {}°: {:.2f}%'.format(position + 1, rate * 100.0))

        print('The attack fails: {:.2f}%'.format(attack_ranking_stats['failure'] * 100.0))
        print()

    print()
    print('===One vs One Comparison===')

    victory_matrix = result_dataset.pairwise_comparison()

    for winner, loser_dict in victory_matrix.items():
        for loser, rate in loser_dict.items():
            print('{} beats {}: {:.2f}%'.format(winner, loser, rate * 100.0))

    if kwargs['save_to'] is not None:
        utils.save_zip(result_dataset, kwargs['save_to'])
