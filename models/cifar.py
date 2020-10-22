import logging

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from torch_utils import MaskedReLU

logger = logging.getLogger(__name__)

model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}

class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        logger.debug('CIFAR features: {}'.format(self.features))
        logger.debug('CIFAR classifier: {}'.format(self.classifier))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

"""def cifar10(n_channel=128, num_classes=10, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=num_classes)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model"""

# Preso da https://github.com/locuslab/convex_adversarial/blob/master/examples/problems.py

# 6144 convoluzionali + 100 MLP = 6244
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

def cifar10_micro_2(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        MaskedReLU([8, 16, 16]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        MaskedReLU([16, 8, 8]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*8*8,50),
        MaskedReLU([50]) if masked_relu else nn.ReLU(),
        nn.Linear(50, num_classes)
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

# 768 convoluzionali + 150 MLP = 918
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

# 1536 convoluzionali + 300 MLP = 1836
def cifar10_medium(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=4, padding=1),
        MaskedReLU([16, 8, 8]) if masked_relu else nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        MaskedReLU([32, 4, 4]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512, 200),
        MaskedReLU([200]) if masked_relu else nn.ReLU(),
        nn.Linear(200, 100),
        MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

# 3072 convoluzionali + 300 MLP = 3372
def cifar10_large(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=1),
        MaskedReLU([8, 16, 16]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 16, 3, stride=2, padding=1),
        MaskedReLU([16, 8, 8]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16*8*8, 200),
        MaskedReLU([200]) if masked_relu else nn.ReLU(),
        nn.Linear(200, 100),
        MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

def cifar10_xs(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 3, 5, stride=4, padding=0),
        MaskedReLU([3, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(147, num_classes)
    )

def cifar10_xs_bn(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 3, 5, stride=4, padding=0),
        MaskedReLU([3, 7, 7]) if masked_relu else nn.ReLU(),
        nn.BatchNorm2d(3),
        nn.Flatten(),
        nn.Linear(147, num_classes)
    )

def cifar10_xs_2(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(392, num_classes)
    )

def cifar10_xs_2_bn(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Flatten(),
        nn.Linear(392, num_classes)
    )

def cifar10_xs_3(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=4, padding=0),
        MaskedReLU([16, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(784, num_classes)
    )

def cifar10_xs_3_bn(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=4, padding=0),
        MaskedReLU([16, 7, 7]) if masked_relu else nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Flatten(),
        nn.Linear(784, num_classes)
    )

def cifar10_xs_4(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 15, 15]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 16, 3, stride=2, padding=0),
        MaskedReLU([16, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(784, num_classes)
    )

def cifar10_xs_5(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 4, 3, stride=2, padding=0),
        MaskedReLU([4, 15, 15]) if masked_relu else nn.ReLU(),
        nn.Conv2d(4, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(392, num_classes)
    )

def cifar10_xs_6(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 16, 3, stride=2, padding=0),
        MaskedReLU([16, 3, 3]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(144, num_classes)
    )

def cifar10_xs_7(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 15, 15]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(392, num_classes)
    )

def cifar10_xs_8(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 15, 15]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1800, num_classes)
    )

def cifar10_xs_9(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=4, padding=0),
        MaskedReLU([32, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1568, num_classes)
    )

def cifar10_xs_10(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4, padding=0),
        MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 3, 3]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(72, num_classes)
    )

def cifar10_xs_11(masked_relu, num_classes):
    return torch.nn.Sequential(
        nn.Conv2d(3, 4, 5, stride=4, padding=0),
        MaskedReLU([4, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Conv2d(4, 8, 3, stride=2, padding=0),
        MaskedReLU([8, 3, 3]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(72, num_classes)
    )

def cifar10(architecture, masked_relu, n_channel=128, num_classes=10, pretrained=True):
    """cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    classifier = nn.Linear(n_channel * 8, num_classes)

    model = nn.Sequential(*layers, nn.Flatten(), classifier)"""
    #model = cifar10_wong_small(masked_relu)
    #model = cifar10_wong_large()
    #model = cifar10_micro_2(masked_relu)

    if architecture == 'extra_small':
        model = cifar10_xs(masked_relu, num_classes)
    elif architecture == 'extra_small_2':
        model = cifar10_xs_2(masked_relu, num_classes)
    elif architecture == 'extra_small_bn':
        model = cifar10_xs_bn(masked_relu, num_classes)
    elif architecture == 'extra_small_2_bn':
        model = cifar10_xs_2_bn(masked_relu, num_classes)
    elif architecture == 'extra_small_3':
        model = cifar10_xs_3(masked_relu, num_classes)
    elif architecture == 'extra_small_3_bn':
        model = cifar10_xs_3_bn(masked_relu, num_classes)
    elif architecture == 'extra_small_4':
        model = cifar10_xs_4(masked_relu, num_classes)
    elif architecture == 'extra_small_5':
        model = cifar10_xs_5(masked_relu, num_classes)
    elif architecture == 'extra_small_6':
        model = cifar10_xs_6(masked_relu, num_classes)
    elif architecture == 'extra_small_7':
        model = cifar10_xs_7(masked_relu, num_classes)
    elif architecture == 'extra_small_8':
        model = cifar10_xs_8(masked_relu, num_classes)
    elif architecture == 'extra_small_9':
        model = cifar10_xs_9(masked_relu, num_classes)
    elif architecture == 'extra_small_10':
        model = cifar10_xs_10(masked_relu, num_classes)
    elif architecture == 'extra_small_11':
        model = cifar10_xs_11(masked_relu, num_classes)
    elif architecture == 'small':
        model = cifar10_small(masked_relu, num_classes)
    elif architecture == 'medium':
        model = cifar10_medium(masked_relu, num_classes)
    elif architecture == 'large':
        model = cifar10_large(masked_relu, num_classes)
    else:
        raise ValueError('Architecture "{}" not supported for CIFAR10.'.format(architecture))

    if pretrained:
        # TODO: Organizzare file
        state_dict = torch.load('./cifar10.pth')
        model.load_state_dict(state_dict)

    return model

def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar100'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

