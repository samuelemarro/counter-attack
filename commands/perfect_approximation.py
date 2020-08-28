import click

import adversarial_dataset as ad
import parsing

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('perfect_distance_dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('approximate_distance_dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
@click.option('--from-perfect-adversarial-dataset', is_flag=True, help='Compute the perfect distance dataset from an adversarial dataset.')
@click.option('--from-approximate-adversarial-dataset', is_flag=True, help='Compute the approximate distance dataset from an adversarial dataset.')
def perfect_approximation(**kwargs):
    perfect_distance_dataset = parsing.get_dataset(kwargs['domain'], kwargs['perfect_distance_dataset'], allow_standard=False)

    if kwargs['from_perfect_adversarial_dataset']:
        perfect_distance_dataset = perfect_distance_dataset.to_distance_dataset()
    elif isinstance(perfect_distance_dataset, ad.AdversarialDataset):
        raise click.BadArgumentUsage('Expected a distance dataset as perfect distance dataset, got an adversarial dataset. '
                                        'If this is intentional, use --from-perfect-adversarial-dataset .')
    
    approximate_distance_dataset = parsing.get_dataset(kwargs['domain'], kwargs['approximate_distance_dataset'], allow_standard=False)

    if kwargs['from_approximate_adversarial_dataset']:
        approximate_distance_dataset = approximate_distance_dataset.to_distance_dataset()
    elif isinstance(approximate_distance_dataset, ad.AdversarialDataset):
        raise click.BadArgumentUsage('Expected a distance dataset as approximate distance dataset, got an adversarial dataset. '
                                        'If this is intentional, use --from-approximate-adversarial-dataset .')

    if perfect_distance_dataset.num_samples < approximate_distance_dataset.num_samples:
        raise click.BadArgumentUsage('The perfect distance dataset contains fewer samples than the approximate one.')


    failed_attacks = 0
    approximate_index = 0

    for perfect_index in range(perfect_distance_dataset.num_samples):
        perfect_image, perfect_distance = perfect_distance_dataset.dataset[perfect_index]
        approximate_image, approximate_distance = approximate_distance_dataset.dataset[approximate_index]

    

# TODO: Implementare e trovare un nome migliore