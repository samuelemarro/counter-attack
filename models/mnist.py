import logging

import torch.nn as nn

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

def mnist_b2(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 200),
        torch_utils.MaskedReLU([200]) if masked_relu else nn.ReLU(),
        nn.Linear(200, num_classes)
    )

def mnist_b3(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 4, 5, stride=4, padding=0),
        torch_utils.MaskedReLU([4, 6, 6]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(144, num_classes)
    )

def mnist_b4(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 100),
        torch_utils.MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, 100),
        torch_utils.MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

def mnist_b5(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 64),
        torch_utils.MaskedReLU([64]) if masked_relu else nn.ReLU(),
        nn.Linear(64, 64),
        torch_utils.MaskedReLU([64]) if masked_relu else nn.ReLU(),
        nn.Linear(64, num_classes)
    )

def mnist_b6(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 4, 5, stride=3, padding=0),
        torch_utils.MaskedReLU([4, 8, 8]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256, num_classes)
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
    elif architecture == 'b2':
        model = mnist_b2(masked_relu, num_classes)
    elif architecture == 'b3':
        model = mnist_b3(masked_relu, num_classes)
    elif architecture == 'b4':
        model = mnist_b4(masked_relu, num_classes)
    elif architecture == 'b5':
        model = mnist_b5(masked_relu, num_classes)
    elif architecture == 'b6':
        model = mnist_b6(masked_relu, num_classes)
    elif architecture == 'c':
        model = mnist_c(masked_relu, num_classes)
    else:
        raise ValueError(
            f'Architecture "{architecture}" not supported for MNIST.')

    return model
