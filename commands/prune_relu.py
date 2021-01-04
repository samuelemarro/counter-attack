import logging

import click
import numpy as np
import torch
import torch.nn as nn

import parsing
import tests
import torch_utils

logger = logging.getLogger(__name__)

# Recursively converts ReLU into ReLUCounter
def recursive_counter(sequential):
    layers = []
    for layer in sequential:
        if isinstance(layer, nn.Sequential):
            layers.append(recursive_counter(layer))
        elif isinstance(layer, nn.ReLU):
            layers.append(torch_utils.ReLUCounter())
        else:
            layers.append(layer)

    return nn.Sequential(*layers)

# Recursively converts ReLUCounter into MaskedReLU
def recursive_converter(sequential, num_samples_threshold):
    total_relus = 0
    replaced_relus = 0
    converted_layers = []
    for layer in sequential:
        if isinstance(layer, nn.Sequential):
            converted_layer, layer_total_relus, layer_replaced_relus = recursive_converter(layer, num_samples_threshold)

            converted_layers.append(converted_layer)
            total_relus += layer_total_relus
            replaced_relus += layer_replaced_relus
        elif isinstance(layer, torch_utils.ReLUCounter):
            zero_mask = layer.negative_counter >= num_samples_threshold
            linear_mask = layer.positive_counter >= num_samples_threshold
            assert zero_mask.shape == linear_mask.shape

            masked_relu = torch_utils.MaskedReLU(zero_mask.shape)
            masked_relu.always_zero.data = zero_mask
            masked_relu.always_linear.data = linear_mask

            converted_layers.append(masked_relu)
            total_relus += np.prod(zero_mask.shape)
            replaced_relus += len(torch.nonzero(zero_mask | linear_mask))
        else:
            converted_layers.append(layer)

    return nn.Sequential(*converted_layers), total_relus, replaced_relus

# TODO: Debuggare

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('original_state_dict_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.argument('threshold', type=float)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
    help='The batch size of the dataset.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--cpu-threads', type=click.IntRange(1, None, False), default=None,
    help='The number of PyTorch CPU threads. If unspecified, the default '
          'number is used (usually the number of cores).')
@click.option('--start', type=click.IntRange(0, None), default=0,
    help='The first index (inclusive) of the dataset that will be used.')
@click.option('--stop', type=click.IntRange(0, None), default=None,
    help='The last index (exclusive) of the dataset that will be used. If unspecified, defaults to '
         'the dataset size.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
def prune_relu(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])
    
    device = kwargs['device']

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['original_state_dict_path'], True, False, load_weights=True)
    model.eval()
    model.to(device)

    if kwargs['threshold'] < 0 or kwargs['threshold'] > 1:
        raise ValueError('Threshold must be between 0 and 1 (inclusive).')

    if not isinstance(model, nn.Sequential):
        raise ValueError('This command only works with sequential networks.')

    if kwargs['dataset'] == 'std:test':
        logger.warn('This command is recommended to be used with non-test datasets.')

    if kwargs['threshold'] < 0.5:
        logger.warn('By using a threshold smaller than 0.5, a lot of unstable ReLUs will be treated as stable. '
                    'Is this intentional?')

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], dataset_edges=(kwargs['start'], kwargs['stop']))
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    counter_model = recursive_counter(model)

    num_samples = 0

    for images, _ in dataloader:
        images = images.to(device)
        num_samples += len(images)
        counter_model(images)

    num_samples_threshold = int(num_samples * kwargs['threshold'])

    converted_model, total_relus, replaced_relus = recursive_converter(counter_model, num_samples_threshold)

    print(f'Replaced {replaced_relus} ReLUs out of {total_relus}.')

    torch.save(converted_model.state_dict(), kwargs['save_to'])
