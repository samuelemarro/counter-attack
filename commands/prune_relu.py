import logging

import click
import numpy as np
import torch
import torch.nn as nn

import parsing
import pruning
import tests
import torch_utils
import utils

logger = logging.getLogger(__name__)

# TODO: Debuggare

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('attacks', callback=parsing.ParameterList(parsing.supported_attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('adversarial_ratio', type=float)
@click.argument('epsilon', type=float)
@click.argument('original_state_dict_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('threshold', type=float)
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
              help='The batch size of the dataset.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--cpu-threads', type=click.IntRange(1, None, False), default=None,
              help='The number of PyTorch CPU threads. If unspecified, the default '
              'number is used (usually the number of cores).')
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='default_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
              'attack configuration.')
@click.option('--start', type=click.IntRange(0, None), default=0,
              help='The first index (inclusive) of the dataset that will be used.')
@click.option('--stop', type=click.IntRange(0, None), default=None,
              help='The last index (exclusive) of the dataset that will be used. If unspecified, defaults to '
              'the dataset size.')
@click.option('--seed', type=int, default=None,
              help='The seed for random generation. If unspecified, the current time is used as seed.')
@click.option('--deterministic', is_flag=True,
              help='If passed, all computations except random number generation are deterministic (but slower).')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
def prune_relu(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['deterministic']:
        if kwargs['seed'] is None:
            logger.warning('Determinism is enabled, but no seed has been provided.')

        utils.enable_determinism()

    logger.debug('Running attack command with kwargs %s.', kwargs)

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])

    if kwargs['seed'] is not None:
        utils.set_seed(kwargs['seed'])

    if kwargs['adversarial_ratio'] <= 0 or kwargs['adversarial_ratio'] > 1:
        raise click.BadArgumentUsage(
            'adversarial_ratio', 'adversarial_ratio must be between 0 (exclusive) and 1 (inclusive).')

    device = kwargs['device']

    model = parsing.parse_model(kwargs['domain'], kwargs['architecture'],
                              kwargs['original_state_dict_path'], True, False, False, load_weights=True)
    model.eval()
    model.to(device)

    if kwargs['threshold'] < 0 or kwargs['threshold'] > 1:
        raise ValueError('Threshold must be between 0 and 1 (inclusive).')

    if not isinstance(model, nn.Sequential):
        raise ValueError('This command only works with sequential networks.')

    if kwargs['dataset'] == 'std:test':
        logger.warning(
            'This command is recommended to be used with non-test datasets.')

    if kwargs['threshold'] < 0.5:
        logger.warning('By using a threshold smaller than 0.5, a lot of unstable ReLUs will be treated as stable. '
                    'Is this intentional?')

    dataset = parsing.parse_dataset(kwargs['domain'], kwargs['dataset'],
                                    dataset_edges=(kwargs['start'], kwargs['stop']))
    dataloader = torch.utils.data.DataLoader(
        dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    attack_pool = parsing.parse_attack_pool(
        kwargs['attacks'], kwargs['domain'], kwargs['p'], 'training', model, attack_config, device)

    converted_model, total_relus, replaced_relus = pruning.prune_relu(
        model, dataloader, attack_pool, kwargs['adversarial_ratio'],
        kwargs['epsilon'], kwargs['threshold'], device)

    print(f'Replaced {replaced_relus} ReLUs out of {total_relus}.')

    torch.save(converted_model.state_dict(), kwargs['save_to'])
