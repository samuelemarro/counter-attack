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
@click.argument('attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='default_attack_configuration.cfg', show_default=True)
@click.option('--keep-misclassified', is_flag=True)
@click.option('--max-samples', type=click.IntRange(1, None), default=None)
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--show', type=click.IntRange(1, None), default=None)
def attack(**kwargs):
    if kwargs['state_dict_path'] is None:
        logger.info('No state dict path provided. Using pretrained model.')

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], max_samples=kwargs['max_samples'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    attack_pool = parsing.get_attack_pool(kwargs['attacks'], kwargs['domain'], kwargs['p'], 'standard', model, attack_config)

    p = kwargs['p']
    
    adversarial_dataset = tests.attack_test(model, attack_pool, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, None)
    distances = adversarial_dataset.distances.numpy()

    success_rate = adversarial_dataset.attack_success_rate
    median_distance = np.median(distances)
    average_distance = np.average(distances)

    print('Success Rate: {:.2f}%'.format(success_rate * 100.0))
    print('Median Distance: {}'.format(median_distance))
    print('Average Distance: {}'.format(average_distance))

    if kwargs['save_to'] is not None:
        utils.save_zip(adversarial_dataset, kwargs['save_to'])

    if kwargs['show'] is not None:
        utils.show_images(adversarial_dataset.genuines, adversarial_dataset.adversarials, limit=kwargs['show'], model=model)