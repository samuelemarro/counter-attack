import parsing
import torch
import utils
import logging
import numpy as np
import tests

logger = logging.getLogger(__name__)

kwargs = {
    'domain' : 'cifar10',
    'dataset' : 'std:test',
    'state_dict_path' : 'trained-models/classifiers/cifar10-mini-500-l1.pth',
    'p' : np.inf,
    'max_samples' : 10,
    'batch_size' : 10,
    'attack_config_file' : 'default_attack_configuration.cfg',
    'attacks' : ['carlini'],
    'device' : 'cuda'
}

model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
model.eval()

dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], max_samples=kwargs['max_samples'])
dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

attack_type = 'standard'

attack_pool = parsing.get_attack_pool(kwargs['attacks'], kwargs['domain'], kwargs['p'], attack_type, model, attack_config, early_rejection_threshold=None)

p = kwargs['p']

adversarial_dataset = tests.attack_test(model, attack_pool, dataloader, p, True, kwargs['device'], attack_config, kwargs, None, blind_trust=False)
adversarial_dataset.print_stats()
