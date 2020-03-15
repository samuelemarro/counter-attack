import logging

import click
import numpy as np
import torch

import parsing
import tests
import utils

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('attacks', callback=parsing.ParameterList(parsing.supported_attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='default_attack_configuration.cfg', show_default=True)
@click.option('--keep-misclassified', is_flag=True)
@click.option('--as-defense', is_flag=True)
@click.option('--early-rejection', type=float, default=None)
@click.option('--max-samples', type=click.IntRange(1, None), default=None)
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--show', type=click.IntRange(1, None), default=None)
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True)
def attack(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], max_samples=kwargs['max_samples'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    attack_type = 'defense' if kwargs['as_defense'] else 'standard'

    attack_pool = parsing.get_attack_pool(kwargs['attacks'], kwargs['domain'], kwargs['p'], attack_type, model, attack_config, early_rejection_threshold=kwargs['early_rejection'])

    p = kwargs['p']
    
    adversarial_dataset = tests.attack_test(model, attack_pool, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, None)
    adversarial_dataset.print_stats()

    if kwargs['save_to'] is not None:
        utils.save_zip(adversarial_dataset, kwargs['save_to'])

    if kwargs['show'] is not None:
        utils.show_images(adversarial_dataset.genuines, adversarial_dataset.adversarials, limit=kwargs['show'], model=model)