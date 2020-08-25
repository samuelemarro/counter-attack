import numpy as np

input_shape = (5, 6)
filter_shape = (3, 3)
input_ = np.array(range(1,np.prod(input_shape)+1)).reshape(input_shape)
filter_ = np.ones(filter_shape)

target = 63

padding_y = 0
padding_x = 0

for i in range(input_.shape[0]):
    for j in range(input_.shape[1]):
        try:
            subinput = input_[i:i+filter_.shape[0], j:j+filter_.shape[1]]
            output = np.sum(subinput * filter_)

            if output == target:
                print('i: {} j: {}'.format(i, j))
        except:
            pass
print('End')