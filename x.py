import tensorflow as tf
import numpy as np

input_shape = [1, 5, 5, 1]
filter_shape = [2, 3, 1, 1]

# order='F' makes NumPy use column-major order, which is the same used by
# Julia
a = np.array(range(1, np.prod(input_shape) + 1)).reshape(input_shape, order='F')
a = tf.convert_to_tensor(a)

filter_ = np.ones(filter_shape)
stride = 1

# Standard padding types
padding = 'VALID'
# padding = 'SAME'

# Padding in the form of [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
#padding = [[0, 0], [1, 2], [3, 4], [0, 0]]

output = tf.nn.conv2d(a, filter_, stride, padding)

# This is the shape that is used to reshape raw_output to
# true_output
print(output.shape)

print('============')

# In the unit test, raw_output is transposed before being used.
# I followed the same convention, but this means that I first have to
# transpose Tensorflow's output so that the unit test will re-transpose it
# back to its original shape
raw_output = output.numpy().transpose()
print(raw_output)

# With all the extra dimensions, raw_output looks awful, so I
# squeeze it:
print(raw_output.squeeze())
print('============')

# To make it easier to copy to the unit tests, here's a small script:
for row in raw_output.squeeze():
    converted_row = [str(elem) for elem in row]
    print('\t'.join(converted_row), end=',\n')

# Depending on your console settings, some additional formatting might
# be necessary