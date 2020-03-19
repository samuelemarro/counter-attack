import logging

import click
import numpy as np
import torch

import adversarial_dataset as ad
import parsing
import tests
import utils

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('attacks', callback=parsing.ParameterList(parsing.supported_attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the state-dict file of the model. If None, a pretrained model will be used (if available).')
@click.option('--from-genuine', default=None,
    help='The source of the genuine dataset. If unspecified, no genuine dataset is used.')
@click.option('--from-adversarial', default=None,
    help='The source of the adversarial dataset. If unspecified, no adversarial dataset is used.')
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
    help='The batch size of the dataset.')
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='default_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
    'attack configuration.')
@click.option('--keep-misclassified', is_flag=True,
    help='If passed, the attack is also run on the images that were misclassified by the base model.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--max-samples', type=click.IntRange(1, None), default=None,
    help='The maximum number of images that are loaded from the dataset. '
         'If unspecified, all images are loaded.')
@click.option('--seed', type=int, default=None,
    help='The seed for random generation. If unspecified, the current time is used as seed.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
def distance_dataset(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['seed'] is not None:
        torch.manual_seed(kwargs['seed'])
    
    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    attack_pool = parsing.get_attack_pool(kwargs['attacks'], kwargs['domain'], kwargs['p'], 'standard', model, attack_config)

    p = kwargs['p']

    if kwargs['from_genuine'] is None and kwargs['from_adversarial'] is None:
        raise RuntimeError('At least one among --from-genuine and --from-adversarial must be provided.')

    images = []
    distances = []

    if kwargs['from_genuine'] is not None:
        genuine_dataset = parsing.get_dataset(kwargs['domain'], kwargs['from_genuine'], max_samples=kwargs['max_samples'])
        genuine_loader = torch.utils.data.DataLoader(genuine_dataset, kwargs['batch_size'], shuffle=False)
        genuine_result_dataset = tests.attack_test(model, attack_pool, genuine_loader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, None)

        images += list(genuine_result_dataset.genuines)
        distances += list(genuine_result_dataset.distances)

    if kwargs['from_adversarial'] is not None:
        adversarial_dataset = parsing.get_dataset(kwargs['domain'], kwargs['from_adversarial'], allow_standard=False, max_samples=kwargs['max_samples'])

        # Get the labels for the adversarial samples
        adversarial_dataset = utils.create_label_dataset(model, adversarial_dataset.adversarials, kwargs['batch_size'])

        adversarial_loader = torch.utils.data.DataLoader(adversarial_dataset, kwargs['batch_size'], shuffle=False)
        adversarial_result_dataset = tests.attack_test(model, attack_pool, adversarial_loader, p, False, kwargs['device'], attack_config, kwargs, None)

        images += list(adversarial_result_dataset.genuines)
        distances += list(adversarial_result_dataset.distances)

    images = torch.stack(images)
    distances = torch.stack(distances)

    final_dataset = ad.AdversarialDistanceDataset(images, distances)

    utils.save_zip(final_dataset, kwargs['save_to'])