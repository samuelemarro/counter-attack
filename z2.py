import numpy as np
import torch
import tensorflow as tf

r = 7

a = torch.arange(50).reshape(10, 5)
a2 = torch.unsqueeze(a, 2).repeat(1, 1, r).numpy()

b = a.numpy()
b2 = tf.tile(tf.expand_dims(b, 2), [1, 1, r]).numpy()

print(a2)
print(b2)
print(a2 - b2)
print(np.max(np.abs(a2 - b2)))