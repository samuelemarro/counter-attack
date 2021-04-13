import logging

import numpy as np
import torch
import torch.nn as nn

import torch_utils

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
