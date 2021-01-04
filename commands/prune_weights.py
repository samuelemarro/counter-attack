import click
import logging
import numpy as np
import torch

import parsing

logger = logging.getLogger(__name__)


@click.command()
@click.argument('domain')
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('original_state_dict_path')
@click.option('--masked-relu', is_flag=True,
              help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument('threshold', type=float)
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
def prune_weights(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'],
                              kwargs['original_state_dict_path'], True, kwargs['masked_relu'], load_weights=True)

    all_parameters = 0
    prunable_parameters = 0

    threshold = kwargs['threshold']

    with torch.no_grad():
        for p in model.parameters():
            if p.dtype == torch.bool:
                continue

            all_parameters += np.prod(list(p.shape))
            below_threshold = torch.abs(p) < threshold
            prunable_parameters += len(torch.nonzero(below_threshold))
            p[below_threshold] = 0.0

    logger.info(
        f'Pruned {prunable_parameters} out of {all_parameters} parameters.')

    torch.save(model.state_dict(), kwargs['save_to'])
