import logging

import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

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

# TODO: Fornirlo come MNIST-mini?
def mnist(input_dims=784, n_hiddens=[24,24], num_classes=10, pretrained=None):
    if pretrained:
        raise NotImplementedError
    
    layers = [nn.Flatten(), nn.Linear(input_dims, n_hiddens[0]), nn.ReLU()]

    for i in range(len(n_hiddens) - 1):
        layers.append(nn.Linear(n_hiddens[i], n_hiddens[i+1]))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(n_hiddens[-1], num_classes))

    return nn.Sequential(*layers)