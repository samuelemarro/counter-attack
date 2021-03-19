import logging

import numpy as np
import torch
import torch.nn as nn

import utils

logger = logging.getLogger(__name__)

def split_batch(x, minibatch_size):
    num_minibatches = int(np.ceil(x.shape[0] / float(minibatch_size)))

    minibatches = []

    for minibatch_id in range(num_minibatches):
        minibatch_begin = minibatch_id * minibatch_size
        minibatch_end = (minibatch_id + 1) * minibatch_size

        minibatches.append(x[minibatch_begin:minibatch_end])

    return minibatches

class Normalisation(nn.Module):
    def __init__(self, mean, std, num_channels=3):
        super().__init__()
        self.mean = torch.from_numpy(
            np.array(mean).reshape((num_channels, 1, 1)))
        self.std = torch.from_numpy(
            np.array(std).reshape((num_channels, 1, 1)))

    def forward(self, x):
        assert x.shape[1] == self.mean.shape[0] == self.std.shape[0]

        mean = self.mean.to(x)
        std = self.std.to(x)
        return (x - mean) / std

# Modular version of torch.squeeze()
class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, self.dim)

# A ReLU module where some ReLU calls are replaced with a fixed behaviour
# (zero or linear)
class MaskedReLU(nn.Module):
    def __init__(self, mask_shape):
        super().__init__()
        self.always_linear = nn.Parameter(torch.zeros(
            mask_shape, dtype=torch.bool), requires_grad=False)
        self.always_zero = nn.Parameter(torch.zeros(
            mask_shape, dtype=torch.bool), requires_grad=False)

    def forward(self, x):
        # Note: We do not use actual masking because
        # that would require using boolean indexing, which
        # causes a CUDA synchronization (causing major slowdowns)

        if self.training:
            logger.warning('MaskedReLU is not designed for training.')

        output = torch.relu(x)

        # always_zero masking
        output = utils.fast_boolean_choice(output, 0, self.always_zero, reshape=False)

        # always_linear masking
        output = utils.fast_boolean_choice(output, x, self.always_linear, reshape=False)

        return output


class ReLUCounter(nn.ReLU):
    def __init__(self):
        super().__init__()
        self.positive_counter = None
        self.negative_counter = None

    def forward(self, x):
        if self.training:
            logger.warning('ReLUCounter is not designed for training.')

        if self.positive_counter is None:
            self.positive_counter = torch.zeros(
                x.shape[1:], dtype=torch.long, device=x.device)
            self.negative_counter = torch.zeros(
                x.shape[1:], dtype=torch.long, device=x.device)

        positive = (x > 0).long().sum(dim=0)
        negative = (x < 0).long().sum(dim=0)

        assert positive.shape == self.positive_counter
        assert negative.shape == self.negative_counter

        self.positive_counter += positive
        self.negative_counter += negative

        return torch.relu(x)


def unpack_sequential(module, ignore=None):
    if ignore is None:
        ignore = []

    layers = []
    for layer in module:
        # If the layer is of a type to be ignored, skip it
        if any([isinstance(layer, t) for t in ignore]):
            continue

        if isinstance(layer, nn.Sequential):
            layers += unpack_sequential(layer)
        else:
            layers.append(layer)

    return layers

def disable_model_gradients(model):
    restore_list = []
    for param in model.parameters():
        restore_list.append((param.requires_grad, param.grad))
        param.requires_grad = False
        param.grad = None

    return restore_list

def restore_model_gradients(model, restore_list):
    parameters = list(model.parameters())
    assert len(parameters) == len(restore_list)

    for param, restore in zip(parameters, restore_list):
        requires_grad, grad = restore
        param.requires_grad = requires_grad
        param.grad = grad