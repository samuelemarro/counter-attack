import json
import sys
sys.path.append('.')

import numpy as np
from numpy import random

import parsing

config = {}

for dataset_name in ['cifar10', 'mnist']:
    dataset = parsing.parse_dataset(dataset_name, 'std:test')

    config[dataset_name] = {
        'architectures' : [],
        'indices' : []
    }

    indices_by_label = {}

    for i in range(10):
        indices_by_label[i] = []

    for index, (_, label) in enumerate(dataset):
        indices_by_label[label].append(index)
    
    for i in range(10):
        indices = indices_by_label[i]
        random.shuffle(indices)
        config[dataset_name]['indices'].append(indices)

with open('job_config.json', 'w') as f:
    json.dump(config, f)