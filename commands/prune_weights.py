import logging
import pathlib

import click
import torch

import parsing
import pruning
import utils

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('original_state_dict_path')
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument('threshold', type=float)
@click.option('--masked-relu', is_flag=True,
              help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--deterministic', is_flag=True,
              help='If passed, all computations except random number generation are deterministic (but slower).')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
def prune_weights(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['deterministic']:
        utils.enable_determinism()

    model = parsing.parse_model(kwargs['domain'], kwargs['architecture'],
                                kwargs['original_state_dict_path'], False,
                                kwargs['masked_relu'], False, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    pruned_count, parameter_count = pruning.prune_weights(model, kwargs['threshold'])

    logger.info(
        'Pruned %s out of %s parameters.', pruned_count, parameter_count)

    save_to = kwargs['save_to']
    pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_to)
