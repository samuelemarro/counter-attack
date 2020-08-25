import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Test:
# Immagine dispari/pari
# Kernel dispari/pari
# Stride 1/2/come kernel
# Note: Tensorflow (in Python) is row-major while Julia
# is column-major. This means that in Tensorflow, in
# order to obtain the same input as in Julia, you should
# first fill a 6x5 matrix in row-major fashion and then
# transpose the result

input_shape = [1, 5, 5, 2]
filter_shape = [3, 3, 2, 1]

filter_shape = [filter_shape[3], filter_shape[2], filter_shape[1], filter_shape[0]]

a = np.array(range(1, np.prod(input_shape) + 1), dtype=np.float)
transposed_input_shape = [input_shape[0], input_shape[3], input_shape[2], input_shape[1]]

#a = a.reshape(*input_shape)
a = a.reshape(*transposed_input_shape)

print(a.shape)

# Trasforma in Torch-compatible
#a = a.transpose([0, 3, 1, 2])

#a = a.transpose([0, 3, 2, 1])
a = torch.from_numpy(a)



filter_ = torch.from_numpy(np.ones(filter_shape))
stride = 2
padding = (1, 1) # Nota: Se l'immagine è trasposta, anche il padding deve essere trasposto!
in_channels = filter_shape[2]
out_channels = filter_shape[3]


#convolution = nn.Conv2d(in_channels, out_channels, (filter_shape[0], filter_shape[1]), stride, padding)

output = F.conv2d(a, filter_, torch.from_numpy(np.array([0])).long(), stride,
                        padding, 1, 1)

output = output.cpu().detach().int().numpy()

# Transforma in standard-compatible
#a = a.transpose([0, ])


# Convert to channel-last
output = output.transpose([0, 2, 3, 1])

print(output.shape)


print(output.flatten())