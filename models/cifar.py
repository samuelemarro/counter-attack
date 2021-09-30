import logging

from collections import OrderedDict

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
        nn.Conv2d(3, 16, 5, stride=4, padding=0),
        MaskedReLU([16, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(784, num_classes)
    )

def cifar10_b2(masked_relu, num_classes):
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


def cifar10_wong_small(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        MaskedReLU([16, 16, 16]) if masked_relu else nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        MaskedReLU([32, 8, 8]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8, 100),
        MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

# TODO: masked_relu


def cifar10_wong_large():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*8*8, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )

def cifar10(architecture, masked_relu, num_classes=10, pretrained=None):
    if pretrained:
        raise NotImplementedError()

    if architecture == 'a':
        model = cifar10_a(masked_relu, num_classes)
    elif architecture == 'b':
        model = cifar10_b(masked_relu, num_classes)
    elif architecture == 'b2':
        model = cifar10_b2(masked_relu, num_classes)
    elif architecture == 'c':
        model = cifar10_c(masked_relu, num_classes)
    elif architecture == 'wong_small':
        model = cifar10_wong_small(masked_relu, num_classes)
    elif architecture == 'wong_large':
        model = cifar10_wong_large()
    else:
        raise ValueError(
            f'Architecture "{architecture}" not supported for CIFAR10.')

    return model
