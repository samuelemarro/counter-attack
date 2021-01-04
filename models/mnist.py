import logging

import torch.nn as nn
from collections import OrderedDict

import torch_utils

logger = logging.getLogger(__name__)


def mnist_a(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 100),
        torch_utils.MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )


def mnist_b(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 24),
        torch_utils.MaskedReLU([24]) if masked_relu else nn.ReLU(),
        nn.Linear(24, 24),
        torch_utils.MaskedReLU([24]) if masked_relu else nn.ReLU(),
        nn.Linear(24, num_classes)
    )


def mnist_c(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 8, 5, stride=4, padding=0),
        torch_utils.MaskedReLU([8, 6, 6]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(288, num_classes)
    )


def mnist(architecture, masked_relu, num_classes=10, pretrained=None):
    if pretrained:
        raise NotImplementedError

    if architecture == 'a':
        model = mnist_a(masked_relu, num_classes)
    elif architecture == 'b':
        model = mnist_b(masked_relu, num_classes)
    elif architecture == 'c':
        model = mnist_c(masked_relu, num_classes)
    else:
        raise ValueError(
            f'Architecture "{architecture}" not supported for MNIST.')

    return model
