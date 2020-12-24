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

def cifar10_wong_small(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        MaskedReLU([16, 16, 16]) if masked_relu else nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        MaskedReLU([32, 8, 8]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8,100),
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
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )

def cifar10_small(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=1),
        MaskedReLU([8, 8, 8]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 16, 3, stride=2, padding=1),
        MaskedReLU([16, 4, 4]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256, 100),
        MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, 50),
        MaskedReLU([50]) if masked_relu else nn.ReLU(),
        nn.Linear(50, num_classes)
    )

def cifar10_x1(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        MaskedReLU([16, 16, 16]) if masked_relu else nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        MaskedReLU([32, 8, 8]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, num_classes)
    )

def cifar10_x2(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=4, padding=0),
        MaskedReLU([16, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Conv2d(16, 16, 3, stride=2, padding=0),
        MaskedReLU([16, 3, 3]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(144, num_classes)
    )

def cifar10_x3(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 15, 15]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(392, num_classes)
    )

def cifar10_x4(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 12, 5, stride=4, padding=0),
        MaskedReLU([12, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Conv2d(12, 12, 3, stride=2, padding=0),
        MaskedReLU([12, 3, 3]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(108, num_classes)
    )

def cifar10_x5(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(392, 100),
        nn.Linear(100, num_classes)
    )

def cifar10_x6(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(392, 100),
        MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

def cifar10_x7(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=4, padding=0),
        MaskedReLU([16, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(784, 100),
        MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

def cifar10_x8(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 12, 3, stride=2, padding=0),
        MaskedReLU([12, 15, 15]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2700, num_classes)
    )

def cifar10_x9(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 500),
        MaskedReLU([500]) if masked_relu else nn.ReLU(),
        nn.Linear(500, num_classes)
    )

def cifar10_x10(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, 200),
        MaskedReLU([200]) if masked_relu else nn.ReLU(),
        nn.Linear(200, num_classes)
    )

def cifar10_x11(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 3, 3]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(72, 72),
        MaskedReLU([72]) if masked_relu else nn.ReLU(),
        nn.Linear(72, num_classes)
    )

def cifar10(architecture, masked_relu, n_channel=128, num_classes=10, pretrained=True):
    if architecture == 'a':
        model = cifar10_a(masked_relu, num_classes)
    elif architecture == 'b':
        model = cifar10_b(masked_relu, num_classes)
    elif architecture == 'c':
        model = cifar10_c(masked_relu, num_classes)
    elif architecture == 'wong_small':
        model = cifar10_wong_small(masked_relu, num_classes)
    elif architecture == 'wong_large':
        model = cifar10_wong_large()
    elif architecture == 'small':
        model = cifar10_small(masked_relu, num_classes)
    elif architecture == 'x1':
        model = cifar10_x1(masked_relu, num_classes)
    elif architecture == 'x2':
        model = cifar10_x2(masked_relu, num_classes)
    elif architecture == 'x3':
        model = cifar10_x3(masked_relu, num_classes)
    elif architecture == 'x4':
        model = cifar10_x4(masked_relu, num_classes)
    elif architecture == 'x5':
        model = cifar10_x5(masked_relu, num_classes)
    elif architecture == 'x6':
        model = cifar10_x6(masked_relu, num_classes)
    elif architecture == 'x7':
        model = cifar10_x7(masked_relu, num_classes)
    elif architecture == 'x8':
        model = cifar10_x8(masked_relu, num_classes)
    elif architecture == 'x9':
        model = cifar10_x9(masked_relu, num_classes)
    elif architecture == 'x10':
        model = cifar10_x10(masked_relu, num_classes)
    elif architecture == 'x11':
        model = cifar10_x11(masked_relu, num_classes)
    else:
        raise ValueError('Architecture "{}" not supported for CIFAR10.'.format(architecture))

    if pretrained:
        # TODO: Organizzare file
        state_dict = torch.load('./cifar10.pth')
        model.load_state_dict(state_dict)

    return model