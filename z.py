import numpy as np
import torch
import torch.nn as nn

from julia import Main
Main.eval('include("C:/Users/Samuele/source/MIPVerify.jl/src/MIPVerify.jl")')

image_height = 32
image_width = 40
in_channels = 10
out_channels = 1
kernel_size = 3
stride = 1

errors = []

for image_height in [20, 31, 32]:
    for image_width in [20, 32, 32]:
        for in_channels in [1, 2, 10]:
            for out_channels in [x for x in [1, 2, 10] if x <= in_channels]:
                for kernel_size in [1, 2, 3, (2, 3)]:
                    for stride in [1, 2, 3]:
                        for padding in [0, (0, 0), 1, (1, 1), 2, (1, 2)]:
                            input_image = torch.rand([1, in_channels, image_height, image_width])
                            convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

                            if isinstance(convolution.stride, tuple):
                                stride = convolution.stride[0]
                            else:
                                stride = convolution.stride

                            filter_ = convolution.weight.cpu().detach().numpy()
                            # Transpose the filter to match MIPVerify
                            filter_ = np.transpose(filter_, [2, 3, 1, 0])

                            bias = convolution.bias.cpu().detach().numpy()

                            if padding is None:
                                mip_padding = Main.MIPVerify.valid
                            else:
                                mip_padding = padding

                            mip_convolution = Main.MIPVerify.Conv2d(filter_, bias, stride, padding)


                            torch_output = convolution(input_image).cpu().detach().numpy()
                            mip_output = Main.MIPVerify.conv2d(input_image.cpu().detach().numpy().transpose([0, 2, 3, 1]), mip_convolution).transpose([0, 3, 1, 2])

                            error = np.sum(np.abs(torch_output - mip_output))
                            if error > 1e-2:
                                print('Image height: {}, width: {}, in_channels: {}, out_channels: {}, kernel_size: {}, stride: {}'.format(image_height, image_width, in_channels, out_channels, kernel_size, stride))
                                print(torch_output.shape)
                                print(mip_output.shape)

                                print(error)
                            errors.append(error)
print(max(errors))