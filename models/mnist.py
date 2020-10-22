import logging

import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

import torch_utils

logger = logging.getLogger(__name__)

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)
        logger.debug('MLP: {}'.format(self.model))

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

"""def mnist(input_dims=784, n_hiddens=[256, 256], num_classes=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, num_classes)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model"""

# 48
def mnist_small(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 24),
        torch_utils.MaskedReLU([24]) if masked_relu else nn.ReLU(),
        nn.Linear(24, 24),
        torch_utils.MaskedReLU([24]) if masked_relu else nn.ReLU(),
        nn.Linear(24, num_classes)
    )

def mnist_small_2(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 48),
        torch_utils.MaskedReLU([48]) if masked_relu else nn.ReLU(),
        nn.Linear(48, 48),
        torch_utils.MaskedReLU([48]) if masked_relu else nn.ReLU(),
        nn.Linear(48, num_classes)
    )

def mnist_small_3(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 36),
        torch_utils.MaskedReLU([36]) if masked_relu else nn.ReLU(),
        nn.Linear(36, 48),
        torch_utils.MaskedReLU([48]) if masked_relu else nn.ReLU(),
        nn.Linear(48, num_classes)
    )

def mnist_small_4(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 100),
        torch_utils.MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

# 200
def mnist_medium(masked_relu, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 100),
        torch_utils.MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, 100),
        torch_utils.MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

def mnist_medium_2(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 8, 5, stride=4, padding=0),
        torch_utils.MaskedReLU([8, 6, 6]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(288, num_classes)
    )

def mnist_medium_3(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 16, 5, stride=4, padding=0),
        torch_utils.MaskedReLU([16, 6, 6]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576, num_classes)
    )

# 392 convoluzionali + 300 MLP = 692
def mnist_large(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 8, 5, stride=4, padding=1),
        torch_utils.MaskedReLU([8, 7, 7]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(392, 200),
        torch_utils.MaskedReLU([200]) if masked_relu else nn.ReLU(),
        nn.Linear(200, 100),
        torch_utils.MaskedReLU([100]) if masked_relu else nn.ReLU(),
        nn.Linear(100, num_classes)
    )

def mnist_large_2(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2, padding=0),
        torch_utils.MaskedReLU([4, 13, 13]) if masked_relu else nn.ReLU(),
        nn.Conv2d(4, 8, 3, stride=2, padding=0),
        torch_utils.MaskedReLU([8, 6, 6]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(288, num_classes)
    )

def mnist_large_3(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, stride=2, padding=0),
        torch_utils.MaskedReLU([8, 13, 13]) if masked_relu else nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=2, padding=0),
        torch_utils.MaskedReLU([8, 6, 6]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(288, num_classes)
    )

def mnist_large_4(masked_relu, num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 16, 5, stride=4, padding=0),
        torch_utils.MaskedReLU([8, 6, 6]) if masked_relu else nn.ReLU(),
        nn.Conv2d(16, 8, 3, stride=2, padding=0),
        torch_utils.MaskedReLU([8, 2, 2]) if masked_relu else nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32, num_classes)
    )

# TODO: Fornirlo come MNIST-mini?
def mnist(architecture, masked_relu, num_classes=10, pretrained=None):
    if pretrained:
        raise NotImplementedError

    if architecture == 'small':
        model = mnist_small(masked_relu, num_classes)
    elif architecture == 'small_2':
        model = mnist_small_2(masked_relu, num_classes)
    elif architecture == 'small_3':
        model = mnist_small_3(masked_relu, num_classes)
    elif architecture == 'small_4':
        model = mnist_small_4(masked_relu, num_classes)
    elif architecture == 'medium':
        model = mnist_medium(masked_relu, num_classes)
    elif architecture == 'medium_2':
        model = mnist_medium_2(masked_relu, num_classes)
    elif architecture == 'medium_3':
        model = mnist_medium_3(masked_relu, num_classes)
    elif architecture == 'large':
        model = mnist_large(masked_relu, num_classes)
    elif architecture == 'large_2':
        model = mnist_large_2(masked_relu, num_classes)
    elif architecture == 'large_3':
        model = mnist_large_3(masked_relu, num_classes)
    elif architecture == 'large_4':
        model = mnist_large_4(masked_relu, num_classes)
    else:
        raise ValueError('Architecture "{}" not supported for MNIST.'.format(architecture))
    
    return model