import os

from torch.nn import parameter
import parsing
import logging
import torch
import tests
import numpy as np
import utils
import json
import os.path
from pathlib import Path
import argparse


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('domain')
parser.add_argument('architecture')
parser.add_argument('count', type=int)

args = parser.parse_args()

domain = args.domain
architecture = args.architecture
count = args.count

def custom_accuracy(domain, architecture, path):
    model = parsing.get_model(domain, architecture, path, True, False, load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(domain, 'std:test')

    dataloader = torch.utils.data.DataLoader(dataset, 50, shuffle=False)
    
    accuracy = tests.accuracy(model, dataloader, 'cuda')

    return accuracy

def tune_mip(domain, architecture, path, pre_path, save_to, p):
    os.system(f'python cli.py tune-mip {domain} {architecture} std:test {p} {save_to} --state-dict-path {path} --pre-adversarial-dataset {pre_path}')

    with open(save_to, 'r') as f:
        text = f.read()

    lines = text.split('  ')
    parameters = dict([(x[0], x[1]) for x in lines])
    # TODO: Non eliminarli?
    parameters.pop('TimeLimit', None)
    parameters.pop('Threads', None)

    return parameters

def create_cfg_file(parameters, save_path):
    with open('attack_configurations/mip_1th_240b_0t_7200s.cfg', 'r') as f:
        cfg = json.load(f)
    for key, value in parameters.items():
        cfg['mip']['all_domains']['all_distances']['all_types'][key] = value
    with open(save_path, 'w') as f:
        json.dump(f)

target_path = f'trained-models/best-classifiers/{domain}-{architecture}.pth'

parsing.set_log_level('info')

if not Path(target_path).exists():
    # TODO: Fornire più controllo?
    os.system(f'train-classifier {domain} {architecture}')

    best_accuracy = -np.inf
    best_es = None

    for es in [5, 10, 25, 50, 100]:
        path = f'trained-models/classifiers/{domain}-{architecture}-es{es}-ftr-1000.pth'
        accuracy = custom_accuracy(domain, architecture, path)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_es = es

    print(f'Best ES: {best_es} (accuracy: {best_accuracy * 100.0}%)')

    best_path = f'trained-models/classifiers/{domain}-{architecture}-es{best_es}-ftr-1000.pth'

    os.system(f'copy {best_path} {target_path}'.replace('/', '\\'))

attack_path = f'adversarial_tests/bim-{domain}-{architecture}-{count}-linf.zip'

if not Path(attack_path).exists():
    os.system(f'python cli.py attack {domain} {architecture} std:test bim linf --max-samples {count} --state-dict-path {target_path}  --save-to {attack_path}')

cfg_path = f'attack_configurations/architecture_specific/mip_1th_240b_0t_7200s_{domain}-{architecture}.cfg'
cfg_f3_path = f'attack_configurations/architecture_specific/mip_1th_240b_0t_7200s_{domain}-{architecture}_f3.cfg'
gurobi_path = f'gurobi-parameter-sets/{domain}-{architecture}.prm'

if not Path(cfg_path).exists():
    parameters = tune_mip(domain, architecture, target_path, attack_path, gurobi_path, 'linf')
    create_cfg_file(parameters, cfg_path)

print(cfg_path)

mip_path = f'mip_results/{domain}-{architecture}-{count}.zip'
mip_f3_path = f'mip_results/{domain}-{architecture}-{count}-f3.zip'

if not Path(cfg_f3_path).exists():
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    reference_dict = cfg['mip']['all_domains']['all_distances']['all_types']
    if 'MIPFocus' not in reference_dict or reference_dict['MIPFocus'] != 3:
        reference_dict['MIPFocus'] = 3
        with open(cfg_f3_path, 'w') as f2:
            json.dump(cfg, f2)

if not Path(mip_path).exists():
    print('Running standard tuned set')
    os.system(f'python cli.py mip {domain} {architecture} std:test linf --max-samples {count} --show {count} --attack-config-file {cfg_path} --state-dict-path {target_path} --pre-adversarial-dataset {attack_path} --save-to {mip_path}')

if Path(cfg_f3_path).exists() and not Path(mip_f3_path).exists():
    print('Running standard tuned set with MIPFocus = 3')
    os.system(f'python cli.py mip {domain} {architecture} std:test linf --max-samples {count} --show {count} --attack-config-file {cfg_f3_path} --state-dict-path {target_path} --pre-adversarial-dataset {attack_path} --save-to {mip_f3_path}')

mip_results = utils.load_zip(mip_path)
print('Standard Convergence')
print('\n'.join([str(x) for x in mip_results.convergence_stats]))

if Path(mip_f3_path).exists():
    mip_f3_results = utils.load_zip(mip_f3_path)
    print('MIPFocus = 3 Convergence:')
    print('\n'.join([str(x) for x in mip_f3_results.convergence_stats]))