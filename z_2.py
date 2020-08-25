import torch
import torch.nn.functional as F
import numpy as np

input_shape = [1, 5, 5, 1]
filter_shape = [2, 3, 1, 1]

# dtype must be float because PyTorch does not support convolutions on integers
a = np.array(range(1, np.prod(input_shape) + 1), dtype=np.float).reshape(input_shape, order='F')

# PyTorch follows the BCHW format (channel-first)
a = a.transpose([0, 3, 1, 2])
a = torch.from_numpy(a)

# MIPVerify follows the filter format:
# (filter_height, filter_width, in_channels, out_channels)
# PyTorch expects the filter format:
# (out_channels, in_channels / groups, kernel_size[0], kernel_size[1])
# since groups=1, filter_height = kernel_size[0] and filter_width = kernel_size[1]
# (out_channels, in_channels, kernel_size[0], kernel_size[1])
torch_filter_shape = [filter_shape[3], filter_shape[2], filter_shape[0], filter_shape[1]]

filter_ = torch.from_numpy(np.ones(torch_filter_shape))

stride = 1
padding = (0, 0)

bias = torch.from_numpy(np.array([0])).long()

output = F.conv2d(a, filter_, bias, stride,
                        padding, 1, 1)

# Convert to a NumPy integer array
output = output.cpu().detach().int().numpy()

# Convert back to BHWC (channel-last)
output = output.transpose([0, 2, 3, 1])

print(output.shape)

# In the unit test, raw_output is transposed before being used.
# I followed the same convention, but this means that I first have to
# transpose PyTorch's output so that the unit test will re-transpose it
# back to its original shape
raw_output = output.transpose()
print(raw_output)
print('============')

# With all the extra dimensions, raw_output looks awful, so I
# squeeze it:
print(raw_output.squeeze())
print('============')

# To make it easier to copy to the unit tests, here's a small script:
for row in raw_output.squeeze():
    converted_row = [str(elem) for elem in row]
    print('\t'.join(converted_row), end=',\n')