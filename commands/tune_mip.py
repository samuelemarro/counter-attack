from logging import log
import click
import logging
from click.types import IntRange
import numpy as np
import torch

import parsing
import utils

from attacks import mip

logger = logging.getLogger(__name__)

# TODO: Tune_limit

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
# TODO: Rimuovere?
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
              help='The batch size of the dataset.')
@click.option('--misclassification-policy', type=click.Choice(parsing.misclassification_policies),
              default='remove', show_default=True, help='The policy that will be applied to deal with '
              'misclassified images.')
@click.option('--pre-adversarial-dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
              help='The path to an adversarial dataset of an attack run on the main dataset. Used to speed up MIP.')
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='default_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
              'attack configuration.')
@click.option('--deterministic', is_flag=True,
              help='If passed, all computations except random number generation are deterministic (but slower).')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
def tune_mip(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['deterministic']:
        utils.enable_determinism()

    if not kwargs['save_to'].endswith('.prm'):
        raise click.BadArgumentUsage(
            'save_to must have a .prm file extension.')

    model = parsing.parse_model(kwargs['domain'], kwargs['architecture'],
                              kwargs['state_dict_path'], True, kwargs['masked_relu'], False, load_weights=True)
    model.eval()

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])
    attack = parsing.parse_attack(
        'mip', kwargs['domain'], kwargs['p'], 'standard', model, attack_config, 'cpu')

    # TODO: model.cpu()?

    if kwargs['pre_adversarial_dataset'] is None:
        pre_adversarial_dataset = None
    else:
        pre_adversarial_dataset = utils.load_zip(
            kwargs['pre_adversarial_dataset'])

        if pre_adversarial_dataset.misclassification_policy != kwargs['misclassification_policy']:
            raise ValueError('The misclassification policy of the pre-adversarial dataset does '
                             'not match the given policy. This can produce incorrent starting points.')

    dataset = parsing.parse_dataset(kwargs['domain'], kwargs['dataset'])

    # The misclassification policy "remove" messes with
    # indexing, so we apply it to the genuine dataset too
    if kwargs['misclassification_policy'] == 'remove':
        all_images = []
        all_true_labels = []
        for start in range(0, len(dataset), kwargs['batch_size']):
            stop = min(start + kwargs['batch_size'], len(dataset))
            indices = range(start, stop)
            images = torch.stack([dataset[i][0] for i in indices])
            true_labels = torch.stack(
                [torch.tensor(dataset[i][1]) for i in indices])
            images, true_labels, _ = utils.apply_misclassification_policy(
                model, images, true_labels, 'remove')
            all_images += list(images)
            all_true_labels += list(true_labels)

        dataset = list(zip(all_images, all_true_labels))

    if pre_adversarial_dataset is None:
        if kwargs['tuning_index'] == -1:
            tuning_index = np.random.randint(len(dataset))
        else:
            tuning_index = kwargs['tuning_index']
        pre_adversarial = None
        pre_image = None
    else:
        successful_indices = [i for i in range(len(
            pre_adversarial_dataset)) if pre_adversarial_dataset.adversarials[i] is not None]
        if kwargs['tuning_index'] == -1:
            tuning_index = np.random.choice(successful_indices)
        else:
            tuning_index = kwargs['tuning_index']
            if tuning_index not in successful_indices:
                logger.warning('The chosen tuning_index does not have a matching '
                               'pre-adversarial. Ignoring pre-adversarial optimizations.')

        pre_adversarial = pre_adversarial_dataset.adversarials[tuning_index]
        pre_adversarial = pre_adversarial.detach().cpu().numpy()
        pre_image = pre_adversarial_dataset.genuines[tuning_index]
        pre_image = pre_image.detach().cpu().numpy()

    image, label = dataset[tuning_index]
    image = image.detach().cpu().numpy()
    label = label.detach().cpu().item()

    if pre_image is not None and np.max(np.abs(image - pre_image)) > 1e-6:
        print(np.max(np.abs(image - pre_image)))
        raise RuntimeError('The pre-adversarial refers to a different genuine. '
                           'This can slow down MIP at best and make it fail at worst. '
                           'Are you sure that you\'re using the correct pre-adversarial dataset?')

    # Implicitly build the MIP model
    # TODO: Non ha senso avere un sistema di retry
    _, adversarial_result = attack.mip_attack(
        image, label, heuristic_starting_point=pre_adversarial)

    jump_model = adversarial_result['Model']

    # Get the Gurobi model
    from julia import JuMP
    from julia import Gurobi
    from julia import Main
    gurobi_model = JuMP.internalmodel(jump_model).inner

    Gurobi.tune_model(gurobi_model)
    Main.model_pointer = gurobi_model
    Main.eval('Gurobi.get_tune_result!(model_pointer, 0)')

    # Save the model
    Gurobi.write_model(gurobi_model, kwargs['save_to'])

    # TODO: Conversion?
