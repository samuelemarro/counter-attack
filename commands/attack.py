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
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the state-dict file of the model. If None, a pretrained model will be used (if available).')
@click.option('--masked-relu', is_flag=True,
    help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
    help='The batch size of the dataset.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--cpu-threads', type=click.IntRange(1, None, False), default=None,
    help='The number of PyTorch CPU threads. If unspecified, the default '
          'number is used (usually the number of cores).')
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='default_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
    'attack configuration.')
@click.option('--misclassification-policy', type=click.Choice(parsing.misclassification_policies),
    default='remove', show_default=True, help='The policy that will be applied to deal with '
    'misclassified images.')
@click.option('--as-defense', is_flag=True,
    help='If passed, the attack is treated as a defense attack.')
@click.option('--early-rejection', type=float, default=None,
    help='The threshold for early rejection. If unspecified, no early rejection is performed.')
@click.option('--blind-trust', is_flag=True)
@click.option('--start', type=click.IntRange(0, None), default=0,
    help='The first index (inclusive) of the dataset that will be used.')
@click.option('--stop', type=click.IntRange(0, None), default=None,
    help='The last index (exclusive) of the dataset that will be used. If unspecified, defaults to '
         'the dataset size.')
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help='The path to the file where the test results will be saved (as a dataset). If unspecified, '
         'no dataset is saved.')
@click.option('--seed', type=int, default=None,
    help='The seed for random generation. If unspecified, the current time is used as seed.')
@click.option('--show', type=click.IntRange(1, None), default=None,
    help='The number of adversarials to be shown. If unspecified, no adversarials are shown.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
def attack(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])

    if kwargs['seed'] is not None:
        torch.manual_seed(kwargs['seed'])

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, kwargs['masked_relu'], load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], dataset_edges=(kwargs['start'], kwargs['stop']))
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    attack_type = 'defense' if kwargs['as_defense'] else 'standard'

    attack_pool = parsing.get_attack_pool(kwargs['attacks'], kwargs['domain'], kwargs['p'], attack_type, model, attack_config)

    p = kwargs['p']

    if kwargs['blind_trust']:
        logger.warn('Blind trust is activated. This means that the success of the attack will NOT be checked.')
    
    adversarial_dataset = tests.attack_test(model, attack_pool, dataloader, p, kwargs['misclassification_policy'], kwargs['device'], kwargs, attack_config, dataset.start, dataset.stop, None, blind_trust=kwargs['blind_trust'])
    adversarial_dataset.print_stats()

    if kwargs['save_to'] is not None:
        utils.save_zip(adversarial_dataset, kwargs['save_to'])

    if kwargs['show'] is not None:
        utils.show_images(adversarial_dataset.genuines, adversarial_dataset.adversarials, limit=kwargs['show'], model=model)