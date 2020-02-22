import logging
import pathlib

import advertorch.bpda
import click
import ignite
import numpy as np
import torch

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

# TODO: Supporto per punto di partenza da state_dict?

@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('epochs', type=click.IntRange(1, None))
@click.argument('save_path', type=click.Path(exists=False, file_okay=True, dir_okay=False))
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

    save_path = kwargs['save_path']
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

# TODO: Che dati usare per addestrare l'approssimatore?
# TODO: Messaggi di errore più accurati se non passi --is-adversarial-dataset

@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('epochs', type=click.IntRange(1, None))
@click.argument('save_path', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--from-adversarial-dataset', is_flag=True)
@add_options(training_options)
def train_approximator(**kwargs):
    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=False, as_detector=True)
    model.train()

    train_dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], allow_standard=False)

    if kwargs['from_adversarial_dataset']:
        train_dataset = train_dataset.to_distance_dataset()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=kwargs['shuffle'])

    val_dataloader = None
    if kwargs['validation_dataset'] is not None:
        val_dataset = parsing.get_dataset(kwargs['domain'], kwargs['validation_dataset'], allow_standard=False)

        if kwargs['from_adversarial_dataset']:
            val_dataset = val_dataset.to_distance_dataset()

        # There's no point in shuffling the validation dataset
        val_dataloader = torch.utils.data.DataLoader(val_dataset, kwargs['batch_size'], shuffle=False)

    loss = torch.nn.MSELoss()
    optimiser = parsing.get_optimiser(kwargs['optimiser'], model.parameters(), kwargs)

    torch_utils.train(model, train_dataloader, optimiser, loss, kwargs['epochs'], kwargs['device'], val_loader=val_dataloader)

    save_path = kwargs['save_path']
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

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
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--show', type=click.IntRange(0, None), default=0, show_default=True)
def attack(**kwargs):
    if kwargs['state_dict_path'] is None:
        logger.info('No state dict path provided. Using pretrained model.')

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    attack_pool = parsing.get_attack_pool(kwargs['attacks'], kwargs['domain'], kwargs['p'], 'standard', model, attack_config)

    p = kwargs['p']
    
    adversarial_dataset = tests.attack_test(model, attack_pool, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, False)
    distances = adversarial_dataset.distances.numpy()

    success_rate = adversarial_dataset.attack_success_rate
    median_distance = np.median(distances)
    average_distance = np.average(distances)

    print('Success Rate: {:.2f}%'.format(success_rate * 100.0))
    print('Median Distance: {}'.format(median_distance))
    print('Average Distance: {}'.format(average_distance))

    if kwargs['save_to'] is not None:
        utils.save_zip(adversarial_dataset, kwargs['save_to'])

    if kwargs['show'] > 0:
        utils.show_images(kwargs['show'], adversarial_dataset.genuines, adversarial_dataset.adversarials)

# Supporto per metrica diversa?
@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('counter_attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('evasion_attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('threshold', type=float)
@click.argument('substitute_architectures', callback=parsing.ParameterList(parsing.architectures))
@click.argument('substitute_state_dict_paths', callback=parsing.ParameterList())
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='default_attack_configuration.cfg', show_default=True)
@click.option('--keep-misclassified', is_flag=True)
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
def evasion_test(**kwargs):
    if kwargs['state_dict_path'] is None:
        logger.info('No state dict path provided. Using pretrained model.')

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])

    p = kwargs['p']

    counter_attack_names = kwargs['counter_attacks']
    substitute_architectures = kwargs['substitute_architectures']
    substitute_state_dict_paths = kwargs['substitute_state_dict_paths']

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

    
    defended_model = detectors.NormalisedDetectorModel(model, detector, -kwargs['threshold'])

    evasion_pool = parsing.get_attack_pool(kwargs['evasion_attacks'], kwargs['domain'], kwargs['p'], 'evasion', model, attack_config, defended_model=defended_model)

    adversarial_dataset = tests.attack_test(defended_model, evasion_pool, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, True)
    distances = adversarial_dataset.distances.numpy()

    success_rate = adversarial_dataset.attack_success_rate
    median_distance = np.median(distances)
    average_distance = np.average(distances)

    print('Success Rate: {:.2f}%'.format(success_rate * 100.0))
    print('Median Distance: {}'.format(median_distance))
    print('Average Distance: {}'.format(average_distance))

    if kwargs['save_to'] is not None:
        utils.save_zip(adversarial_dataset, kwargs['save_to'])

# Nota: In questo test, il threshold indica "se togli l'attacco corrispondente, quanto deve ottenere la detector pool per rifiutare?"

@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('thresholds', callback=parsing.ParameterList(cast_to=float))
@click.argument('substitute_architectures', callback=parsing.ParameterList(parsing.architectures))
@click.argument('substitute_state_dict_paths', callback=parsing.ParameterList())
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='default_attack_configuration.cfg', show_default=True)
@click.option('--keep-misclassified', is_flag=True)
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
def cross_validation(**kwargs):
    if kwargs['state_dict_path'] is None:
        logger.info('No state dict path provided. Using pretrained model.')

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])
    p = kwargs['p']

    attack_names = kwargs['attacks']
    thresholds = kwargs['thresholds']
    substitute_architectures = kwargs['substitute_architectures']
    substitute_state_dict_paths = kwargs['substitute_state_dict_paths']

    if len(thresholds) == 1:
        thresholds = len(attack_names) * [thresholds[0]]

    if len(substitute_architectures) == 1:
        substitute_architectures = len(attack_names) * [substitute_architectures[0]]

    if len(thresholds) != len(attack_names):
        raise click.BadArgumentUsage('thresholds must be either one value or as many values as the number of attacks.')

    if len(substitute_architectures) != len(attack_names):
        raise click.BadArgumentUsage('substitute_architectures must be either one value or as many values as the number of attacks.')

    if len(substitute_state_dict_paths) != len(attack_names):
        raise click.BadArgumentUsage('substitute_state_dict_paths must be as many values as the number of attacks.')

    test_names = []
    evasion_attacks = []
    defended_models = []

    for i in range(len(attack_names)):
        # The counter attacks, their substitute architectures and their state dict paths
        # are all the passed values except for the one that will act as an evasion attack

        evasion_attack_name = attack_names[i]
        counter_attack_names = [x for j, x in enumerate(attack_names) if j != i]

        ca_substitute_architectures = [x for j, x in enumerate(substitute_architectures) if j != i]
        ca_substitute_state_dict_paths = [x for j, x in enumerate(substitute_state_dict_paths) if j != i]

        threshold = thresholds[i]
        
        detector = parsing.get_detector_pool(counter_attack_names, kwargs['domain'], kwargs['p'], 'standard', model, attack_config, kwargs['device'],
        substitute_architectures=ca_substitute_architectures, substitute_state_dict_paths=ca_substitute_state_dict_paths)

        defended_model = detectors.NormalisedDetectorModel(model, detector, -threshold)

        evasion_attack = parsing.get_attack(evasion_attack_name, kwargs['domain'], kwargs['p'], 'evasion', model, attack_config, defended_model=defended_model)

        test_name = '{} vs {}'.format(evasion_attack_name, counter_attack_names)

        test_names.append(test_name)
        evasion_attacks.append(evasion_attack)
        defended_models.append(defended_model)

    logger.info('Tests:\n{}'.format('\n'.join(test_names)))

    evasion_dataset = tests.multiple_evasion_test(model, test_names, evasion_attacks, defended_models, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, True)

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

    if kwargs['save_to'] is not None:
        utils.save_zip(evasion_dataset, kwargs['save_to'])

@main.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('attacks', callback=parsing.ParameterList(parsing.attacks))
@click.argument('p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance)
@click.argument('thresholds', callback=parsing.ParameterList(cast_to=float))
@click.argument('substitute_architectures', callback=parsing.ParameterList(parsing.architectures))
@click.argument('substitute_state_dict_paths', callback=parsing.ParameterList())
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False), default='default_attack_configuration.cfg', show_default=True)
@click.option('--keep-misclassified', is_flag=True)
@click.option('--save-to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
def attack_matrix(**kwargs):
    if kwargs['state_dict_path'] is None:
        logger.info('No state dict path provided. Using pretrained model.')

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()
    model.to(kwargs['device'])

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'])
    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)

    attack_config = utils.read_attack_config_file(kwargs['attack_config_file'])
    p = kwargs['p']

    attack_names = kwargs['attacks']
    thresholds = kwargs['thresholds']
    substitute_architectures = kwargs['substitute_architectures']
    substitute_state_dict_paths = kwargs['substitute_state_dict_paths']

    if len(thresholds) == 1:
        thresholds = len(attack_names) * [thresholds[0]]

    if len(substitute_architectures) == 1:
        substitute_architectures = len(attack_names) * [substitute_architectures[0]]

    if len(thresholds) != len(attack_names):
        raise click.BadArgumentUsage('thresholds must be either one value or as many values as the number of attacks.')

    if len(substitute_architectures) != len(attack_names):
        raise click.BadArgumentUsage('substitute_architectures must be either one value or as many values as the number of attacks.')

    if len(substitute_state_dict_paths) != len(attack_names):
        raise click.BadArgumentUsage('substitute_state_dict_paths must be as many values as the number of attacks.')

    test_names = []
    evasion_attacks = []
    defended_models = []

    for evasion_attack_name in attack_names:
        for counter_attack_name, ca_substitute_architecture, ca_substitute_state_dict_path, threshold in zip(attack_names, substitute_architectures, substitute_state_dict_paths, thresholds):
            detector = parsing.get_detector(counter_attack_name, kwargs['domain'], kwargs['p'], 'standard', model, attack_config, kwargs['device'],
            substitute_architecture=ca_substitute_architecture, substitute_state_dict_path=ca_substitute_state_dict_path)

            defended_model = detectors.NormalisedDetectorModel(model, detector, -threshold)

            evasion_attack = parsing.get_attack(evasion_attack_name, kwargs['domain'], kwargs['p'], 'evasion', model, attack_config, defended_model=defended_model)

            test_name = '{} vs {}'.format(evasion_attack_name, counter_attack_name)

            test_names.append(test_name)
            evasion_attacks.append(evasion_attack)
            defended_models.append(defended_model)

    evasion_dataset = tests.multiple_evasion_test(model, test_names, evasion_attacks, defended_models, dataloader, p, not kwargs['keep_misclassified'], kwargs['device'], attack_config, kwargs, True)

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

    if kwargs['save_to'] is not None:
        utils.save_zip(evasion_dataset, kwargs['save_to'])
if __name__  == '__main__':
    main()