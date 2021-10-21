import logging

import torch
import torch.nn as nn

from torch_utils import MaskedReLU

logger = logging.getLogger(__name__)


def cifar10_a(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 15, 15]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1800, num_classes)
    )

def cifar10_b(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 20, 5, stride=4, padding=0),
        MaskedReLU([20, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(980, num_classes)
    )


def cifar10_c(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 3, 3]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(72, num_classes)
    )

def cifar10(architecture, masked_relu, num_classes=10, pretrained=None):
    if pretrained:
        raise NotImplementedError()

    if architecture == 'a':
        model = cifar10_a(masked_relu, num_classes)
    elif architecture == 'b':
        model = cifar10_b(masked_relu, num_classes)
    elif architecture == 'c':
        model = cifar10_c(masked_relu, num_classes)
    else:
        raise ValueError(
            f'Architecture "{architecture}" not supported for CIFAR10.')

    return model
