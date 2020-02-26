import logging
import pathlib

import advertorch.bpda
import click
import ignite
import numpy as np
import torch

# To avoid name mismatches
import adversarial_dataset as ad

import custom_attacks
import detectors
import parsing
import tests
import torch_utils
import utils

logger = logging.getLogger(__name__)

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options

training_options = [
    click.option('--optimiser', type=click.Choice(['adam', 'sgd']), default='adam', show_default=True,
        help='The optimiser that will be used for training.'),
    click.option('--learning_rate', type=float, default=1e-3, show_default=True,
        help='The learning rate for the optimiser.'),
    click.option('--weight-decay', type=float, default=0, show_default=True,
        help='The weight decay for the optimiser.'),
    click.option('--adam-betas', nargs=2, type=click.Tuple([float, float]), default=(0.9, 0.999), show_default=True,
        help='The two beta values. Ignored if the optimiser is not \'adam\''),
    click.option('--adam-epsilon', type=float, default=1e-8, show_default=True,
        help='The value of epsilon. Ignored if the optimiser is not \'adam\''),
    click.option('--adam-amsgrad', is_flag=True,
        help='Enables AMSGrad. Ignored if the optimiser is not \'adam\''),
    click.option('--sgd-momentum', type=float, default=0, show_default=True,
        help='The intensity of momentum. Ignored if the optimiser is not \'sgd\''),
    click.option('--sgd-dampening', type=float, default=0, show_default=True,
        help='The intensity of dampening. Ignored if the optimiser is not \'sgd\''),
    click.option('--sgd-nesterov', is_flag=True,
        help='Enables Nesterov Accelerated Gradient. Ignored if the optimiser is not \'adam\''),
    click.option('--validation-dataset', default=None),
    click.option('--shuffle', is_flag=True)
]

@click.group()
def main():
    pass

@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('epochs', type=click.IntRange(1, None))
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@add_options(training_options)
def train_classifier(**kwargs):
    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=False)
    model.train()

    train_dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=kwargs['shuffle'])

    val_dataloader = None
    if kwargs['validation_dataset'] is not None:
        val_dataset = parsing.get_dataset(kwargs['domain'], kwargs['validation_dataset'])
        val_dataloader = torch.utils.data.DataLoader(val_dataset, kwargs['batch_size'], shuffle=False)

    additional_metrics = {'Accuracy' : ignite.metrics.Accuracy()}
    loss = torch.nn.CrossEntropyLoss()
    optimiser = parsing.get_optimiser(kwargs['optimiser'], model.parameters(), kwargs)

    torch_utils.train(model, train_dataloader, optimiser, loss, kwargs['epochs'], kwargs['device'], val_loader=val_dataloader, additional_metrics=additional_metrics)

    save_to = kwargs['save_to']
    pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_to)

# TODO: Dividere in più file?

# TODO: Spiegare che --from-adversarial-dataset usa genuine-adversarial per calcolare la distanza per i genuine

# Nota: keep_misclassified viene ignorato per gli adversarial examples, dato che per definizione vengono misclassificati
@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--from-genuine', default=None)
@click.option('--from-adversarial', default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='default_attack_configuration.cfg', show_default=True)
@click.option('--keep-misclassified', is_flag=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--max-samples', type=click.IntRange(1, None), default=None)
def distance_dataset(**kwargs):
    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    attack_pool = parsing.get_attack_pool(kwargs['attacks'], kwargs['domain'], kwargs['p'], 'standard', model, attack_config)

    p = kwargs['p']

    if kwargs['from_genuine'] is None and kwargs['from_adversarial'] is None:
        raise RuntimeError('At least one among --from-genuine and --from-adversarial must be provided.')

    images = []
    distances = []

    if kwargs['from_genuine'] is not None:
        genuine_dataset = parsing.get_dataset(kwargs['domain'], kwargs['from_genuine'], max_samples=kwargs['max_samples'])
        genuine_loader = torch.utils.data.DataLoader(genuine_dataset, kwargs['batch_size'], shuffle=False)
        genuine_result_dataset = tests.attack_test(model, attack_pool, genuine_loader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, None)

        images += list(genuine_result_dataset.genuines)
        distances += list(genuine_result_dataset.distances)

    if kwargs['from_adversarial'] is not None:
        adversarial_dataset = parsing.get_dataset(kwargs['domain'], kwargs['from_adversarial'], allow_standard=False, max_samples=kwargs['max_samples'])

        # Get the labels for the adversarial samples
        adversarial_dataset = utils.create_label_dataset(model, adversarial_dataset.adversarials, kwargs['batch_size'])

        adversarial_loader = torch.utils.data.DataLoader(adversarial_dataset, kwargs['batch_size'], shuffle=False)
        adversarial_result_dataset = tests.attack_test(model, attack_pool, adversarial_loader, p, False, kwargs['device'], attack_config, kwargs, None)

        images += list(adversarial_result_dataset.genuines)
        distances += list(adversarial_result_dataset.distances)

    images = torch.stack(images)
    distances = torch.stack(distances)

    final_dataset = ad.AdversarialDistanceDataset(images, distances)

    utils.save_zip(final_dataset, kwargs['save_to'])
    
@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('epochs', type=click.IntRange(1, None))
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--from-adversarial-dataset', is_flag=True)
@click.option('--val-from-adversarial-dataset', is_flag=True)
@add_options(training_options)
def train_approximator(**kwargs):
    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=False, as_detector=True)
    model.train()

    train_dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], allow_standard=False)

    if kwargs['from_adversarial_dataset']:
        train_dataset = train_dataset.to_distance_dataset()
    elif isinstance(train_dataset, ad.AdversarialDataset):
        raise click.BadArgumentUsage('Expected a distance dataset as training dataset, got an adversarial dataset. '
                                    'If this is intentional, use --from-adversarial-dataset .')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=kwargs['shuffle'])

    val_dataloader = None
    if kwargs['validation_dataset'] is not None:
        val_dataset = parsing.get_dataset(kwargs['domain'], kwargs['validation_dataset'], allow_standard=False)

        if kwargs['val_from_adversarial_dataset']:
            val_dataset = val_dataset.to_distance_dataset()
        elif isinstance(val_dataset, ad.AdversarialDataset):
            raise click.BadArgumentUsage('Expected a distance dataset as validation dataset, got an adversarial dataset. '
                                        'If this is intentional, use --val-from-adversarial-dataset .')

        # There's no point in shuffling the validation dataset
        val_dataloader = torch.utils.data.DataLoader(val_dataset, kwargs['batch_size'], shuffle=False)

    loss = torch.nn.MSELoss()
    optimiser = parsing.get_optimiser(kwargs['optimiser'], model.parameters(), kwargs)

    torch_utils.train(model, train_dataloader, optimiser, loss, kwargs['epochs'], kwargs['device'], val_loader=val_dataloader)

    save_to = kwargs['save_to']
    pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_to)

@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--from-adversarial-dataset', is_flag=True)
def accuracy(**kwargs):
    if kwargs['state_dict_path'] is None:
        logger.info('No state dict path provided. Using pretrained model.')

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'])

    if kwargs['from_adversarial_dataset']:
        dataset = dataset.to_adversarial_training_dataset()
        logger.warning('The accuracy will be computed only on the successful adversarial examples.')

    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)
    
    accuracy = tests.accuracy(model, dataloader, kwargs['device'])

    print('Accuracy: {:.2f}%'.format(accuracy * 100.0))

@main.command()
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

# Supporto per metrica diversa?
@main.command()
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
def evasion_test(**kwargs):
    if kwargs['state_dict_path'] is None:
        logger.info('No state dict path provided. Using pretrained model.')

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
        logger.warn('You are using a positive rejection rejection_threshold. Since Counter-Attack only outputs nonpositive values, '
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
                                        'standard',
                                        model,
                                        attack_config,
                                        kwargs['device'],
                                        substitute_architectures=substitute_architectures,
                                        substitute_state_dict_paths=substitute_state_dict_paths)

    
    defended_model = detectors.NormalisedDetectorModel(model, detector, kwargs['rejection_threshold'])

    evasion_pool = parsing.get_attack_pool(kwargs['evasion_attacks'], kwargs['domain'], kwargs['p'], 'evasion', model, attack_config, defended_model=defended_model)

    adversarial_dataset = tests.attack_test(model, evasion_pool, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, defended_model)
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

# Nota: In questo test, il rejection_threshold indica "se togli l'attacco corrispondente, quanto deve ottenere la detector pool per rifiutare?"

@main.command()
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
        substitute_architectures=ca_substitute_architectures, substitute_state_dict_paths=ca_substitute_state_dict_paths)

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
        distances = adversarial_dataset.distances.numpy()

        success_rate = adversarial_dataset.attack_success_rate
        median_distance = np.median(distances)
        average_distance = np.average(distances)

        print('Success Rate: {:.2f}%'.format(success_rate * 100.0))
        print('Median Distance: {}'.format(median_distance))
        print('Average Distance: {}'.format(average_distance))

        if kwargs['show'] is not None:
            utils.show_images(adversarial_dataset.genuines, adversarial_dataset.adversarials, limit=kwargs['show'], model=model)

# TODO: Perché il tasso di successo è diverso quando il threshold di rifiuto è 50?
# TODO: La CLI è scomoda quando devi passare valori negativi

@main.command()
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
def attack_matrix(**kwargs):
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

    for evasion_attack_name in attack_names:
        for counter_attack_name, ca_substitute_architecture, ca_substitute_state_dict_path, rejection_threshold in zip(attack_names, substitute_architectures, substitute_state_dict_paths, rejection_thresholds):
            detector = parsing.get_detector(counter_attack_name, kwargs['domain'], kwargs['p'], 'standard', model, attack_config, kwargs['device'],
            substitute_architecture=ca_substitute_architecture, substitute_state_dict_path=ca_substitute_state_dict_path)

            defended_model = detectors.NormalisedDetectorModel(model, detector, rejection_threshold)

            evasion_attack = parsing.get_attack(evasion_attack_name, kwargs['domain'], kwargs['p'], 'evasion', model, attack_config, defended_model=defended_model)

            test_name = '{} vs {}'.format(evasion_attack_name, counter_attack_name)

            test_names.append(test_name)
            evasion_attacks.append(evasion_attack)
            defended_models.append(defended_model)

    evasion_dataset = tests.multiple_evasion_test(model, test_names, evasion_attacks, defended_models, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs)

    logger.info('Tests:\n{}'.format('\n'.join(test_names)))

    for test_name in test_names:
        print('Test "{}":'.format(test_name))
        adversarial_dataset = evasion_dataset.to_adversarial_dataset(test_name)
        distances = adversarial_dataset.distances.numpy()

        success_rate = adversarial_dataset.attack_success_rate
        median_distance = np.median(distances)
        average_distance = np.average(distances)

        print('Success Rate: {:.2f}%'.format(success_rate * 100.0))
        print('Median Distance: {}'.format(median_distance))
        print('Average Distance: {}'.format(average_distance))

        if kwargs['show'] is not None:
            utils.show_images(adversarial_dataset.genuines, adversarial_dataset.adversarials, limit=kwargs['show'], model=model)

    if kwargs['save_to'] is not None:
        utils.save_zip(evasion_dataset, kwargs['save_to'])
if __name__  == '__main__':
    main()