import logging

import advertorch
import click
import custom_attacks
import numpy as np
import torch
import torchvision

import cifar10_models
import detectors
import torch_utils
import utils

logger = logging.getLogger(__name__)

domains = ['cifar10']
architectures = ['resnet50']
attacks = ['carlini', 'fgm']
distances = ['l2', 'linf']

def get_model(domain, architecture, state_dict_path, apply_normalisation, load_weights=False, as_detector=False):
    if as_detector:
        num_classes = 1
    if domain == 'cifar10':
        if not as_detector:
            num_classes = 10
        if architecture == 'resnet50':
            pretrained = load_weights and state_dict_path is None
            model = cifar10_models.resnet50(pretrained=pretrained, num_classes=num_classes)
        else:
            raise NotImplementedError('Unsupported architecture {} for domain {}.'.format(architecture, domain))
    else:
        raise NotImplementedError('Unsupported domain {}.'.format(domain))

    if apply_normalisation:
        if domain == 'cifar10':
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
        else:
            raise NotImplementedError('Unsupported normalisation for domain {}.'.format(domain))

        normalisation = torch_utils.Normalisation(mean, std)
        model = torch.nn.Sequential(normalisation, model)

    # Nota: Questo fa sì che i modelli vadano salvati come modello con normalisation
    if load_weights and state_dict_path is not None:
        logger.info('Loading weights from {}'.format(state_dict_path))
        model.load_state_dict(torch.load(state_dict_path))

    return model

def get_dataset(domain, dataset, allow_standard=True):
    matched_dataset = None
    transform = torchvision.transforms.ToTensor()
    if allow_standard:
        if domain == 'cifar10':
            if dataset == 'std:train':
                matched_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
            elif dataset == 'std:test':
                matched_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    
    if matched_dataset is None:
        # 'dataset' non corrisponde a nulla, proviamo ad aprire il file
        try:
            matched_dataset = utils.load_zip(dataset)
        except:
            raise RuntimeError('Could not find a standard dataset or a dataset file "{}".'.format(dataset))

    return matched_dataset

def get_optimiser(optimiser_name, learnable_parameters, options):
    if optimiser_name == 'adam':
        optimiser = torch.optim.Adam(
            learnable_parameters, lr=options['learning_rate'], betas=options['adam_betas'], weight_decay=options['weight_decay'], eps=options['adam_epsilon'], amsgrad=options['adam_amsgrad'])
    elif optimiser_name == 'sgd':
        optimiser = torch.optim.SGD(
            learnable_parameters, lr=options['learning_rate'], momentum=options['sgd_momentum'],
            dampening=options['sgd_dampening'], weight_decay=options['weight_decay'], nesterov=options['sgd_nesterov'])
    else:
        raise ValueError('Optimiser not supported.')

    return optimiser

def get_attack(attack_name, domain, distance_metric, attack_type, model, attack_config, defended_model=None):
    kwargs = attack_config.get_arguments(attack_name, domain, distance_metric, attack_type)

    logger.info('Preparing attack {}, domain {}, distance metric {}, type {} with kwargs: {}'.format(
        attack_name, domain, distance_metric, attack_type, kwargs 
    ))

    if attack_type == 'evasion' and defended_model is None:
        raise ValueError('Evasion attacks require a defended model.')

    if domain == 'cifar10':
        num_classes = 10
    else:
        raise NotImplementedError('Unsupported domain "{}".'.format(domain))

    if attack_name == 'carlini':
        if distance_metric == 'l2':
            if attack_type == 'standard':
                attack = advertorch.attacks.CarliniWagnerL2Attack(model, num_classes, **kwargs)
            elif attack_type == 'evasion':
                # TODO: è la scelta migliore?
                attack = advertorch.attacks.CarliniWagnerL2Attack(defended_model, num_classes + 1, **kwargs)
                attack = custom_attacks.KBestTargetAttackAgainstDetector(model, attack)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, distance_metric, attack_type))
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, distance_metric))
    elif attack_name == 'fgm':
        if distance_metric == 'l2':
            if attack_type == 'standard':
                attack = advertorch.attacks.FGM(model, **kwargs)
            elif attack_type == 'evasion':
                # TODO: è solo un placeholder
                attack = advertorch.attacks.FGM(defended_model, **kwargs)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, distance_metric, attack_type))
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, distance_metric))
    else:
        raise NotImplementedError('Unsupported attack "{}".'.format(attack_name))

    return attack

def get_attack_pool(attack_names, domain, distance_metric, attack_type, model, attack_config, defended_model=None):
    if len(attack_names) == 1:
        return get_attack(attack_names[0], domain, distance_metric, attack_type, model, attack_config, defended_model=defended_model)
    else:
        attacks = []
        for attack_name in attack_names:
            attacks.append(get_attack(attack_name, domain, distance_metric, attack_type, model, attack_config, defended_model=defended_model))

        p = get_p(distance_metric)
        return custom_attacks.AttackPool(attacks, p)

def get_detector(attack_name, domain, distance_metric, attack_type, model, attack_config, device, substitute_architecture=None, substitute_state_dict_path=None):
    model.to(device)
    p = get_p(distance_metric)

    if substitute_architecture is not None:
        assert substitute_state_dict_path is not None

    attack = get_attack(attack_name, domain, distance_metric, attack_type, model, attack_config, defended_model=None)
    detector = detectors.CounterAttackDetector(attack, p)

    if substitute_architecture is not None:
        substitute_detector = get_model(domain, substitute_architecture, substitute_state_dict_path, True, load_weights=True, as_detector=True)

        # The substitute model returns a [batch_size, 1] matrix, while we need a [batch_size] vector
        substitute_detector = torch.nn.Sequential(substitute_detector, torch_utils.Squeeze(1))
        substitute_detector.to(device)

        detector = advertorch.bpda.BPDAWrapper(detector, forwardsub=substitute_detector)

    return detector

# TODO: Controllare/Finire/Usare

def get_detector_pool(attack_names, domain, distance_metric, attack_type, model, attack_config, device, substitute_architectures=None, substitute_state_dict_paths=None):
    p = get_p(distance_metric)

    if substitute_architectures is not None:
        assert len(substitute_architectures) == len(attack_names)
        assert len(substitute_state_dict_paths) == len(attack_names)
    
    detector_pool = []

    for i in range(len(attack_names)):
        substitute_architecture = None
        substitute_state_dict_path = None
        if substitute_architectures is not None:
            substitute_architecture = substitute_architectures[i]
            substitute_state_dict_path = substitute_state_dict_paths[i]

        detector = get_detector(attack_names[i], domain, distance_metric, attack_type, model, attack_config, device, substitute_architecture=substitute_architecture, substitute_state_dict_path=substitute_state_dict_path)
        detector_pool.append(detector)

    if len(detector_pool) == 1:
        return detector_pool[0]
    else:
        return detectors.DetectorPool(detector_pool, p)

_distance_to_p = {'l2': 2, 'linf' : np.inf}

def get_p(distance_metric):
    if distance_metric == 'l2':
        return 2
    elif distance_metric == 'linf':
        return np.inf
    else:
        raise NotImplementedError('Unsupported distance metric "{}."'.format(distance_metric))

class ParameterList:
    def __init__(self, allowed_values=None, cast_to=None):
        self.allowed_values = allowed_values
        self.cast_to = cast_to

    def __call__(self, ctx, param, value):
        value = value.replace('[', '').replace(']', '')
        if ',' in value:
            try:
                parameter_list = [parameter.strip() for parameter in value.split(',')]
            except:
                raise click.BadParameter('Parameter must be in format "[value_1, value_2, ...]" (with quotes, if there are spaces).')
        else:
            parameter_list = [value]

        if self.allowed_values is not None:
            for parameter in parameter_list:
                if parameter not in self.allowed_values:
                    raise click.BadParameter('Unrecognised value "{}".'.format(parameter))

        if self.cast_to is not None:
            parameter_list = [self.cast_to(parameter) for parameter in parameter_list]

        return parameter_list