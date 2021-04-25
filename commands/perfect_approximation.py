import logging

import click
import matplotlib.pyplot as plt
import numpy as np

import adversarial_dataset as ad
import parsing
import utils

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('perfect_distance_dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('approximate_distance_dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
@click.option('--from-perfect-adversarial-dataset', is_flag=True, help='Compute the perfect distance dataset from an adversarial dataset.')
@click.option('--from-approximate-adversarial-dataset', is_flag=True, help='Compute the approximate distance dataset from an adversarial dataset.')
@click.option('--deterministic', is_flag=True,
              help='If passed, all computations except random number generation are deterministic (but slower).')
def perfect_approximation(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['deterministic']:
        utils.enable_determinism()

    perfect_distance_dataset = parsing.parse_dataset(
        kwargs['domain'], kwargs['perfect_distance_dataset'], allow_standard=False)

    if kwargs['from_perfect_adversarial_dataset']:
        perfect_distance_dataset = perfect_distance_dataset.to_distance_dataset(
            failure_value=None)
    elif isinstance(perfect_distance_dataset, ad.AdversarialDataset):
        raise click.BadArgumentUsage('Expected a distance dataset as perfect distance dataset, got an adversarial dataset. '
                                     'If this is intentional, use --from-perfect-adversarial-dataset .')

    approximate_distance_dataset = parsing.parse_dataset(
        kwargs['domain'], kwargs['approximate_distance_dataset'], allow_standard=False)

    if kwargs['from_approximate_adversarial_dataset']:
        approximate_distance_dataset = approximate_distance_dataset.to_distance_dataset(
            failure_value=None)
    elif isinstance(approximate_distance_dataset, ad.AdversarialDataset):
        raise click.BadArgumentUsage('Expected a distance dataset as approximate distance dataset, got an adversarial dataset. '
                                     'If this is intentional, use --from-approximate-adversarial-dataset .')

    if len(perfect_distance_dataset) < len(approximate_distance_dataset):
        raise click.BadArgumentUsage(
            'The perfect distance dataset contains fewer samples than the approximate one.')

    absolute_differences = []
    relative_differences = []

    for (perfect_genuine, perfect_distance), (approximate_genuine, approximate_distance) in zip(perfect_distance_dataset, approximate_distance_dataset):
        # TODO: Sostituire con una costante
        if np.max(np.abs(perfect_genuine - approximate_genuine)) > 1e-5:
            raise click.BadArgumentUsage(
                'Datasets don\'t match (different genuine images).')

        if approximate_distance is None:
            continue

        # TODO: Qui probabilmente serve qualche tolerance
        if approximate_distance < perfect_distance:
            raise click.BadArgumentUsage(
                'Invalid datasets (approximate is better than perfect).')

        absolute_differences.append(approximate_distance - perfect_distance)
        relative_differences.append(
            (approximate_distance - perfect_distance) / perfect_distance)

    absolute_differences = np.array(absolute_differences)
    relative_differences = np.array(relative_differences)

    print('Average absolute difference: {:.3e}'.format(
        np.average(absolute_differences)))
    print('Average relative difference: {:.2f}%'.format(
        np.average(relative_differences) * 100.0))
    print('Minimum absolute difference: {:.3e}'.format(
        np.min(absolute_differences)))
    print('Minimum relative difference: {:.2f}%'.format(
        np.min(relative_differences) * 100.0))
    if len(absolute_differences) > 0:
        bins = []
        total = 0
        step = 1 / 510
        while total < absolute_differences.max():
            bins.append(total)
            total += step
        bins = np.array(bins)
        plt.hist(absolute_differences, bins=bins)
        plt.show()
