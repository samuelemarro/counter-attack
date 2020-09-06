import parsing
import torch
import numpy as np

model = parsing.get_model('cifar10', './trained-models/classifiers/cifar10-micro-250-l1-04.pth', True, load_weights=True)

all_parameters = 0
prunable_parameters = 0

threshold = 1e-3

with torch.no_grad():
    for p in model.parameters():
        all_parameters += np.prod(list(p.shape))
        prunable_parameters += len(torch.nonzero(torch.abs(p) < threshold))
        #p[torch.abs(p) < threshold] = 0.0

    print(prunable_parameters)
    print(all_parameters)

torch.save(model.state_dict(), './trained-models/classifiers/cifar10-micro-250-l1-04-pruned.pth')