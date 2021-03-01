import torch
import torch.nn as nn

import numpy as np

import utils

import logging
logger = logging.getLogger(__name__)


def split_batch(x, minibatch_size):
    num_minibatches = int(np.ceil(x.shape[0] / float(minibatch_size)))

    minibatches = []

    for minibatch_id in range(num_minibatches):
        minibatch_begin = minibatch_id * minibatch_size
        minibatch_end = (minibatch_id + 1) * minibatch_size

        minibatches.append(x[minibatch_begin:minibatch_end])

    return minibatches


class BatchLimitedModel(nn.Module):
    def __init__(self, wrapped_model, batch_size):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.batch_size = batch_size

    def forward(self, x):
        outputs = []

        for minibatch in split_batch(x, self.batch_size):
            outputs.append(self.wrapped_model(minibatch))

        outputs = torch.cat(outputs)

        assert len(outputs) == len(x)

        return outputs


class Normalisation(nn.Module):
    def __init__(self, mean, std, num_channels=3):
        super().__init__()
        self.mean = torch.from_numpy(
            np.array(mean).reshape((num_channels, 1, 1)))
        self.std = torch.from_numpy(
            np.array(std).reshape((num_channels, 1, 1)))

    def forward(self, input):
        mean = self.mean.to(input)
        std = self.std.to(input)
        return (input - mean) / std

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

# TODO: Controlla

# A ReLU module where some ReLU calls are replaced with fixed behaviour
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
        if self.positive_counter is None:
            self.positive_counter = torch.zeros(
                x.shape[1:], dtype=torch.long, device=x.device)
            self.negative_counter = torch.zeros(
                x.shape[1:], dtype=torch.long, device=x.device)

        positive = (x > 0).long().sum(dim=0)
        negative = (x < 0).long().sum(dim=0)

        self.positive_counter += positive
        self.negative_counter += negative

        return torch.relu(x)


def unpack_sequential(module):
    layers = []
    for layer in module:
        if isinstance(layer, nn.Sequential):
            layers += unpack_sequential(layer)
        else:
            layers.append(layer)

    return layers

def split_dataset(original_dataset, val_split, shuffle=True):
    dataset_size = len(original_dataset)
    indices = list(range(dataset_size))
    split_index = int(np.floor(val_split * dataset_size))

    if shuffle:
        np.random.shuffle(indices)

    val_indices, train_indices = indices[:split_index], indices[split_index:]

    train_dataset = IndexedDataset(original_dataset, train_indices)
    val_dataset = IndexedDataset(original_dataset, val_indices)

    return train_dataset, val_dataset