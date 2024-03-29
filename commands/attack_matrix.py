import logging

import click
import torch

import detectors
import parsing
import tests
import utils

logger = logging.getLogger(__name__)

# TODO: La CLI è scomoda quando devi passare valori negativi


@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('attacks', callback=parsing.ParameterList(parsing.supported_attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('rejection_thresholds', callback=parsing.ParameterList(cast_to=float))
@click.argument('substitute_state_dict_paths', callback=parsing.ParameterList())
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
              help='The path to the state-dict file of the model. If None, a pretrained model will be used (if available).')
@click.option('--masked-relu', is_flag=True,
              help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.option('--batch-size', type=click.IntRange(1, None), default=50, show_default=True,
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
@click.option('--deterministic', is_flag=True,
              help='If passed, all computations except random number generation are deterministic (but slower).')
@click.option('--show', type=click.IntRange(1, None), default=None,
              help='The number of adversarials to be shown. If unspecified, no adversarials are shown.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
def attack_matrix(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['deterministic']:
        if kwargs['seed'] is None:
            logger.warning('Determinism is enabled, but no seed has been provided.')

        utils.enable_determinism()

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])

    if kwargs['seed'] is not None:
        utils.set_seed(kwargs['seed'])

    model = parsing.parse_model(kwargs['domain'], kwargs['architecture'],
                              kwargs['state_dict_path'], False, kwargs['masked_relu'], False, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    dataset = parsing.parse_dataset(kwargs['domain'], kwargs['dataset'], dataset_edges=(
        kwargs['start'], kwargs['stop']))
    dataloader = torch.utils.data.DataLoader(
        dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])
    p = kwargs['p']

    attack_names = kwargs['attacks']
    rejection_thresholds = kwargs['rejection_thresholds']
    substitute_state_dict_paths = kwargs['substitute_state_dict_paths']

    if len(rejection_thresholds) == 1:
        rejection_thresholds = len(attack_names) * [rejection_thresholds[0]]

    if len(rejection_thresholds) != len(attack_names):
        raise click.BadArgumentUsage(
            'rejection_thresholds must be either one value or as many values as the number of attacks.')

    if len(substitute_state_dict_paths) != len(attack_names):
        raise click.BadArgumentUsage(
            'substitute_state_dict_paths must be as many values as the number of attacks.')

    if any(rejection_threshold > 0 for rejection_threshold in rejection_thresholds):
        logger.warning('Received a positive rejection threshold. Since Counter-Attack only outputs nonpositive values, '
                    'the detector will never reject an example.')

    test_names = []
    evasion_attacks = []
    defended_models = []

    for evasion_attack_name in attack_names:
        for counter_attack_name, ca_substitute_state_dict_path, rejection_threshold in zip(attack_names, substitute_state_dict_paths, rejection_thresholds):
            detector = parsing.parse_detector(counter_attack_name, kwargs['domain'], kwargs['p'], 'standard', model, attack_config, kwargs['device'],
                                            use_substitute=True, substitute_state_dict_path=ca_substitute_state_dict_path)

            defended_model = detectors.NormalisedDetectorModel(
                model, detector, rejection_threshold)

            evasion_attack = parsing.parse_attack(
                evasion_attack_name, kwargs['domain'], kwargs['p'], 'evasion', model, attack_config, kwargs['device'], defended_model=defended_model, seed=kwargs['seed'])

            test_name = f'{evasion_attack_name} vs {counter_attack_name}'

            test_names.append(test_name)
            evasion_attacks.append(evasion_attack)
            defended_models.append(defended_model)

    evasion_dataset = tests.multiple_evasion_test(model, test_names, evasion_attacks, defended_models, dataloader,
                                                  p, kwargs['misclassification_policy'], kwargs['device'], attack_config, dataset.start, dataset.stop, kwargs)

    logger.info('Tests:\n{}'.format('\n'.join(test_names)))

    for test_name in test_names:
        print(f'Test "{test_name}":')
        adversarial_dataset = evasion_dataset.to_adversarial_dataset(test_name)
        adversarial_dataset.print_stats()

        if kwargs['show'] is not None:
            utils.show_images(adversarial_dataset.genuines,
                              adversarial_dataset.adversarials, limit=kwargs['show'], model=model)

    if kwargs['save_to'] is not None:
        utils.save_zip(evasion_dataset, kwargs['save_to'])
