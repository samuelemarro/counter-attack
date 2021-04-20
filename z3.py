from julia import Base
import numpy as np
import torch
import torch.nn as nn

import attacks.mip as mip
import torch_utils

shape = [3 * 32 * 32]
image_shape = [3, 32, 32]

r = torch.rand(shape)
mask1 = r > 0.7
mask2 = r < 0.2

module = torch_utils.MaskedReLU(shape)
module.always_linear.data = mask1
module.always_zero.data = mask2
module.eval()

module2 = torch_utils.MaskedReLU([3, 32, 32])
module2.eval()
r2 = torch.rand([3, 32, 32])
module2.always_linear.data = r2 <= .8
module2.always_zero.data = r2 > 0.95

module = nn.Sequential(module2, nn.Flatten(),  module)

a = torch.rand(image_shape) - 0.5

converted = mip.sequential_to_mip(module)
#converted = mip.module_to_mip(module)

print(module(a.unsqueeze(0)).shape)

standard_output = module(a.unsqueeze(0))[0].detach().cpu().numpy()

mip_input = a.cpu().numpy()

if len(image_shape) == 3:
    mip_input = mip_input.transpose(1, 2, 0)
    print(mip_input.shape)

mip_output = converted(np.expand_dims(mip_input, 0)).squeeze()

if len(mip_output.shape) == 3:
    mip_output = mip_output.transpose(2, 0, 1)

print('Standard: ', standard_output.shape)
print('MIP output: ', mip_output.shape)

print(standard_output-mip_output)
print(np.max(np.abs(standard_output - mip_output)))