import logging

import click
import numpy as np
import torch

import attacks
import parsing
import tests
import torch_utils
import utils

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the state-dict file of the model. If None, a pretrained model will be used (if available).')
@click.option('--masked-relu', is_flag=True,
    help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
    help='The batch size of the dataset.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--cpu-threads', type=click.IntRange(1, None, False), default=None,
    help='The number of PyTorch CPU threads. If unspecified, the default '
          'number is used (usually the number of cores).')
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='default_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
    'attack configuration.')
@click.option('--misclassification-policy', type=click.Choice(parsing.misclassification_policies),
    default='remove', show_default=True, help='The policy that will be applied to deal with '
    'misclassified images.')
# TODO: Rimuovere?
@click.option('--gurobi-model', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the cached Gurobi model that will be used to run the attack. If unspecified, the model '
         'is built on the fly. Warning: no checks on whether the Gurobi model is the same as the PyTorch model '
         'are performed.')
@click.option('--pre-adversarial-dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to an adversarial dataset of an attack run on the main dataset. Used to speed up MIP.')
@click.option('--start', type=click.IntRange(0, None), default=0,
    help='The first index (inclusive) of the dataset that will be used.')
@click.option('--stop', type=click.IntRange(0, None), default=None,
    help='The last index (exclusive) of the dataset that will be used. If unspecified, defaults to '
         'the dataset size.')
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help='The path to the file where the test results will be saved (as a dataset). If unspecified, '
         'no dataset is saved.')
@click.option('--seed', type=int, default=None,
    help='The seed for random generation. If unspecified, the current time is used as seed.')
@click.option('--show', type=click.IntRange(1, None), default=None,
    help='The number of adversarials to be shown. If unspecified, no adversarials are shown.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
def mip(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])

    if kwargs['seed'] is not None:
        torch.manual_seed(kwargs['seed'])

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, kwargs['masked_relu'], load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], start=kwargs['start'], stop=kwargs['stop'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    if kwargs['pre_adversarial_dataset'] is None:
        pre_adversarial_dataset = None
    else:
        pre_adversarial_dataset = utils.load_zip(kwargs['pre_adversarial_dataset'])

    p = kwargs['p']

    if p == 2:
        metric = 'l2'
    elif np.isposinf(p):
        metric = 'linf'
    else:
        raise NotImplementedError('Unsupported metric "l{}"'.format(p))

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])
    attack_kwargs = attack_config.get_arguments('mip', kwargs['domain'], metric, 'standard')

    attack = attacks.MIPAttack(model, p, False, **attack_kwargs)
    
    mip_dataset = tests.mip_test(model, attack, dataloader, p, kwargs['misclassification_policy'], kwargs['device'], attack_config, kwargs,
                                 pre_adversarial_dataset=pre_adversarial_dataset)
    mip_dataset.print_stats()

    if kwargs['save_to'] is not None:
        utils.save_zip(mip_dataset, kwargs['save_to'])

    if kwargs['show'] is not None:
        utils.show_images(mip_dataset.genuines, mip_dataset.adversarials, limit=kwargs['show'], model=model)