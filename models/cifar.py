import logging

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

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

def cifar10_c(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 3, 3]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(72, num_classes)
    )

def cifar10(architecture, masked_relu, n_channel=128, num_classes=10, pretrained=True):
    if architecture == 'a':
        model = cifar10_a(masked_relu, num_classes)
    elif architecture == 'b':
        model = cifar10_b(masked_relu, num_classes)
    elif architecture == 'c':
        model = cifar10_c(masked_relu, num_classes)
    else:
        raise ValueError('Architecture "{}" not supported for CIFAR10.'.format(architecture))

    if pretrained:
        # TODO: Organizzare file
        state_dict = torch.load('./cifar10.pth')
        model.load_state_dict(state_dict)

    return model