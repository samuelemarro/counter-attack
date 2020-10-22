import parsing
import attacks.mip as mip
import numpy as np
import torch
import torch.nn as nn

#model = parsing.get_model('cifar10', 'trained-models/classifiers/cifar10-micro-250-l1-04-pruned.pth', True, load_weights=True)

def test(perm):

    model = parsing.get_model('mnist', 'trained-models/classifiers/mnist-mini-100.pth', True, False, load_weights=True)

    model.eval()

    mip_model = mip.sequential_to_mip(model)

    torch_image = list(parsing.get_dataset('mnist', 'std:test', max_samples=1))[0][0]

    model.cpu()
    torch_output = model(torch_image.unsqueeze(0)).squeeze().detach().cpu().numpy()

    image = torch_image.cpu().numpy()

    image = image.transpose([1, 2, 0])

    extra_dimension = image.shape[-1] == 1
    image = np.expand_dims(image, 0)

    output = mip_model(image)
    """print(output.shape)
    print(torch_output.shape)
    print(output)
    print(torch_output)
    print(output - torch_output)
    print(np.max(output - torch_output))"""
    if np.max(np.abs(output - torch_output)).item() < 1e-3:
        print('Perm: {} ({})'.format(perm, np.max(np.abs(output - torch_output)).item()))

import itertools
for perm in [
    [3, 2, 4, 1]
]:
    print(perm)
    test(list(perm))