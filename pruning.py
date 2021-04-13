import logging

import numpy as np
import torch
import torch.nn as nn

import torch_utils
import training

logger = logging.getLogger(__name__)

def prune_weights(model, threshold):
    layers = torch_utils.unpack_sequential(model)

    cumulative_pruned_count = 0
    cumulative_parameter_count = 0

    for layer in layers:
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            below_threshold = torch.abs(layer.weight) < threshold
            layer.weight[below_threshold] = 0

            layer_pruned_count = torch.count_nonzero(below_threshold).detach().cpu().item()
            layer_parameter_count = np.prod(below_threshold.shape).item()

            logger.info('Pruned %s parameters out of %s in %s',
                        layer_pruned_count,
                        layer_parameter_count,
                        type(layer).__name__)

            cumulative_pruned_count += layer_pruned_count
            cumulative_parameter_count += layer_parameter_count

    return cumulative_pruned_count, cumulative_parameter_count

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
            converted_layer, layer_total_relus, layer_replaced_relus = recursive_converter(
                layer, num_samples_threshold)

            converted_layers.append(converted_layer)
            total_relus += layer_total_relus
            replaced_relus += layer_replaced_relus
        elif isinstance(layer, torch_utils.ReLUCounter):
            zero_mask = layer.nonpositive_counter >= num_samples_threshold
            linear_mask = layer.positive_counter >= num_samples_threshold
            assert zero_mask.shape == linear_mask.shape
            assert not (zero_mask & linear_mask).any()

            masked_relu = torch_utils.MaskedReLU(zero_mask.shape)
            masked_relu.always_zero.data = zero_mask
            masked_relu.always_linear.data = linear_mask

            converted_layers.append(masked_relu)
            total_relus += np.prod(zero_mask.shape).item()
            replaced_relus += len(torch.nonzero(zero_mask | linear_mask))
        else:
            converted_layers.append(layer)

    return nn.Sequential(*converted_layers), total_relus, replaced_relus

def prune_relu(model, dataloader, attack, attack_ratio, epsilon, threshold, device):
    counter_model = recursive_counter(model)
    counter_model.eval()

    num_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        num_samples += len(images)

        images = training.adversarial_training(images, labels, model, attack, attack_ratio, epsilon)
        counter_model(images)

    num_samples_threshold = num_samples * threshold

    return recursive_converter(counter_model, num_samples_threshold)
