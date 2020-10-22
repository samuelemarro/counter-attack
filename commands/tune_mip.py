import click
import logging
from click.types import IntRange
import numpy as np
import torch

import parsing
import utils

from attacks import mip

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain')
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the state-dict file of the model. If None, a pretrained model will be used (if available).')
@click.option('--tuning-index', type=click.IntRange(-1, None), default=-1, show_default=True,
    help='The index of the image that will be chosen as the reference for tuning. '
         'If -1, a random image is chosen.')
@click.option('--masked-relu', is_flag=True,
    help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='default_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
    'attack configuration.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
def tune_mip(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if not kwargs['save_to'].endswith('.prm'):
        raise click.BadArgumentUsage('save_to must have a .prm file extension.')

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, kwargs['masked_relu'], load_weights=True)

    attack_config = attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])
    attack = parsing.get_attack('mip', kwargs['domain'], kwargs['p'], 'standard', model, attack_config)

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], max_samples=1)
    if kwargs['tuning_index'] == -1:
        tuning_index = np.random.randint(len(dataset))
    else:
        tuning_index = kwargs['tuning_index']
    
    image, label = dataset[tuning_index]
    image = image.detach().cpu().numpy()

    # Implicitly build the MIP model
    _, adversarial_result = attack.mip_attack(image, label)

    jump_model = adversarial_result['Model']

    # Get the Gurobi model
    from julia import JuMP
    from julia import Gurobi
    from julia import Main
    gurobi_model = JuMP.internalmodel(jump_model).inner

    # TODO: Set parameters
    Gurobi.tune_model(gurobi_model)
    Main.model_pointer = gurobi_model
    Main.eval('Gurobi.get_tune_result!(model_pointer, 0)')


    # Save the model
    Gurobi.write_model(gurobi_model, kwargs['save_to'])