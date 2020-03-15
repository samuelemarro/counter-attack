import logging

import click
import numpy as np
import torch

import detectors
import parsing
import tests
import utils

logger = logging.getLogger(__name__)

# Nota: In questo test, il rejection_threshold indica "se togli l'attacco corrispondente, quanto deve ottenere la detector pool per rifiutare?"

# Nota: Gli early_rejection_threshold sono negati

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('rejection_thresholds', callback=parsing.ParameterList(cast_to=float))
@click.argument('substitute_architectures', callback=parsing.ParameterList(parsing.architectures))
@click.argument('substitute_state_dict_paths', callback=parsing.ParameterList())
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='default_attack_configuration.cfg', show_default=True)
@click.option('--keep-misclassified', is_flag=True)
@click.option('--max-samples', type=click.IntRange(1, None), default=None)
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--show', type=click.IntRange(1, None), default=None)
def cross_validation(**kwargs):
    if kwargs['state_dict_path'] is None:
        logger.info('No state dict path provided. Using pretrained model.')

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], max_samples=kwargs['max_samples'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])
    p = kwargs['p']

    attack_names = kwargs['attacks']
    rejection_thresholds = kwargs['rejection_thresholds']
    substitute_architectures = kwargs['substitute_architectures']
    substitute_state_dict_paths = kwargs['substitute_state_dict_paths']

    if len(attack_names) < 2:
        raise click.BadArgumentUsage('attacks must be at least two.')

    if len(rejection_thresholds) == 1:
        rejection_thresholds = len(attack_names) * [rejection_thresholds[0]]

    if len(substitute_architectures) == 1:
        substitute_architectures = len(attack_names) * [substitute_architectures[0]]

    if len(rejection_thresholds) != len(attack_names):
        raise click.BadArgumentUsage('rejection_thresholds must be either one value or as many values as the number of attacks.')

    if len(substitute_architectures) != len(attack_names):
        raise click.BadArgumentUsage('substitute_architectures must be either one value or as many values as the number of attacks.')

    if len(substitute_state_dict_paths) != len(attack_names):
        raise click.BadArgumentUsage('substitute_state_dict_paths must be as many values as the number of attacks.')

    if any(rejection_threshold > 0 for rejection_threshold in rejection_thresholds):
        logger.warn('You are using a positive rejection threshold. Since Counter-Attack only outputs nonpositive values, '
        'the detector will never reject an example.')

    test_names = []
    evasion_attacks = []
    defended_models = []

    for i in range(len(attack_names)):
        # Remove one attack from the pool. This attack will act as the evasion attack

        evasion_attack_name = attack_names[i]
        counter_attack_names = [x for j, x in enumerate(attack_names) if j != i]

        ca_substitute_architectures = [x for j, x in enumerate(substitute_architectures) if j != i]
        ca_substitute_state_dict_paths = [x for j, x in enumerate(substitute_state_dict_paths) if j != i]

        rejection_threshold = rejection_thresholds[i]
        
        detector = parsing.get_detector_pool(counter_attack_names, kwargs['domain'], kwargs['p'], 'standard', model, attack_config, kwargs['device'],
        substitute_architectures=ca_substitute_architectures, substitute_state_dict_paths=ca_substitute_state_dict_paths, early_rejection_threshold=-rejection_threshold)

        defended_model = detectors.NormalisedDetectorModel(model, detector, rejection_threshold)

        evasion_attack = parsing.get_attack(evasion_attack_name, kwargs['domain'], kwargs['p'], 'evasion', model, attack_config, defended_model=defended_model)

        test_name = '{} vs {}'.format(evasion_attack_name, counter_attack_names)

        test_names.append(test_name)
        evasion_attacks.append(evasion_attack)
        defended_models.append(defended_model)

    logger.info('Tests:\n{}'.format('\n'.join(test_names)))

    evasion_dataset = tests.multiple_evasion_test(model, test_names, evasion_attacks, defended_models, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs)

    if kwargs['save_to'] is not None:
        utils.save_zip(evasion_dataset, kwargs['save_to'])

    for test_name in test_names:
        print('Test "{}":'.format(test_name))
        adversarial_dataset = evasion_dataset.to_adversarial_dataset(test_name)
        adversarial_dataset.print_stats()

        if kwargs['show'] is not None:
            utils.show_images(adversarial_dataset.genuines, adversarial_dataset.adversarials, limit=kwargs['show'], model=model)