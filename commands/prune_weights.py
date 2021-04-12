import click
import logging
import numpy as np
import torch

import parsing
import torch_utils
import utils

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain')
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('original_state_dict_path')
@click.option('--masked-relu', is_flag=True,
              help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument('threshold', type=float)
@click.option('--deterministic', is_flag=True,
              help='If passed, all computations except random number generation are deterministic (but slower).')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
def prune_weights(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['deterministic']:
        utils.enable_determinism()

    model = parsing.parse_model(kwargs['domain'], kwargs['architecture'],
                              kwargs['original_state_dict_path'], True, kwargs['masked_relu'], False, load_weights=True)

    # TODO: Debuggare
    simplified_model = torch_utils.unpack_sequential(model, ignore=[torch_utils.Normalisation])

    all_parameters = 0
    prunable_parameters = 0

    threshold = kwargs['threshold']

    with torch.no_grad():
        for param in simplified_model.parameters():
            if param.dtype == torch.bool:
                continue

            all_parameters += np.prod(list(param.shape))
            below_threshold = torch.abs(param) < threshold
            prunable_parameters += len(torch.nonzero(below_threshold))
            param[below_threshold] = 0.0

    logger.info(
        'Pruned %s out of %s parameters.', prunable_parameters, all_parameters)

    torch.save(model.state_dict(), kwargs['save_to'])
