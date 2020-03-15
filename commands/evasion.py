import logging

import click
import numpy as np
import torch

import detectors
import parsing
import tests
import utils

logger = logging.getLogger(__name__)

# Nota: Molto spesso, il threshold di rifiuto influenza la distanza ottenuta anche quando non influenza
# il tasso di successo dell'evasion. Questo è perché cambiare il threshold cambia la loss, cambiando
# quindi il comportamento dell'attacco

# Supporto per metrica diversa?
@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('counter_attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('evasion_attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('rejection_threshold', type=float)
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
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True)
def evasion(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], max_samples=kwargs['max_samples'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    p = kwargs['p']

    counter_attack_names = kwargs['counter_attacks']
    substitute_architectures = kwargs['substitute_architectures']
    substitute_state_dict_paths = kwargs['substitute_state_dict_paths']

    if kwargs['rejection_threshold'] >= 0:
        logger.warn('You are using a positive rejection threshold. Since Counter-Attack only outputs nonpositive values, '
        'the detector will never reject an example.')

    if len(substitute_architectures) == 1:
        substitute_architectures = len(counter_attack_names) * [substitute_architectures[0]]

    if len(substitute_architectures) != len(counter_attack_names):
        raise click.BadArgumentUsage('substitute_architectures must be either one value or as many values as the number of counter attacks.')

    if len(substitute_state_dict_paths) != len(counter_attack_names):
        raise click.BadArgumentUsage('substitute_state_dict_paths must be as many values as the number of counter attacks.')

    detector = parsing.get_detector_pool(counter_attack_names,
                                        kwargs['domain'],
                                        kwargs['p'],
                                        'defense',
                                        model,
                                        attack_config,
                                        kwargs['device'],
                                        substitute_architectures=substitute_architectures,
                                        substitute_state_dict_paths=substitute_state_dict_paths,
                                        early_rejection_threshold=-kwargs['rejection_threshold'])

    
    defended_model = detectors.NormalisedDetectorModel(model, detector, kwargs['rejection_threshold'])

    evasion_pool = parsing.get_attack_pool(kwargs['evasion_attacks'], kwargs['domain'], kwargs['p'], 'evasion', model, attack_config, defended_model=defended_model)

    adversarial_dataset = tests.attack_test(model, evasion_pool, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, defended_model)
    adversarial_dataset.print_stats()

    if kwargs['save_to'] is not None:
        utils.save_zip(adversarial_dataset, kwargs['save_to'])

    if kwargs['show'] is not None:
        utils.show_images(adversarial_dataset.genuines, adversarial_dataset.adversarials, limit=kwargs['show'], model=model)