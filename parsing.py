import logging

import advertorch
import advertorch.bpda
import click
import custom_attacks
import evasion_attacks
import numpy as np
import torch
import torchvision

import standard_attacks
import cifar10_models
import detectors
import torch_utils
import utils

logger = logging.getLogger(__name__)

domains = ['cifar10']
architectures = ['resnet50']
attacks = ['bim', 'carlini', 'deepfool', 'fast_gradient', 'pgd']
attacks_with_binary_search = ['bim', 'fast_gradient', 'pgd']
distances = ['l2', 'linf']

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

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options

def get_model(domain, architecture, state_dict_path, apply_normalisation, load_weights=False, as_detector=False):
    if as_detector:
        num_classes = 1
    
    pretrained = load_weights and state_dict_path is None

    if domain == 'cifar10':
        if not as_detector:
            num_classes = 10
        
        if architecture == 'resnet50':
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

    # Nota: Questo fa sì che i modelli vengano salvati come modello con normalisation
    if load_weights and state_dict_path is not None:
        logger.info('Loading weights from {}'.format(state_dict_path))
        model.load_state_dict(torch.load(state_dict_path))

    return model

def get_dataset(domain, dataset, allow_standard=True, max_samples=None):
    matched_dataset = None
    transform = torchvision.transforms.ToTensor()
    
    if allow_standard:
        if domain == 'cifar10':
            if dataset == 'std:train':
                matched_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
            elif dataset == 'std:test':
                matched_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
    
    if matched_dataset is None:
        # No matches found, try to read it as a file path
        try:
            matched_dataset = utils.load_zip(dataset)
        except:
            raise RuntimeError('Could not find a standard dataset or a dataset file "{}".'.format(dataset))

    if max_samples is not None:
        matched_dataset = torch_utils.FirstNDataset(matched_dataset, max_samples)

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
        raise ValueError('Unsupported optimiser "{}".'.format(optimiser_name))

    return optimiser

def get_attack(attack_name, domain, p, attack_type, model, attack_config, defended_model=None):
    # Convert the float value to its standard name
    if p == 2:
        metric = 'l2'
    elif np.isposinf(p):
        metric = 'linf'
    else:
        raise NotImplementedError('Unsupported metric "l{}"'.format(p))

    kwargs = attack_config.get_arguments(attack_name, domain, metric, attack_type)

    logger.debug('Preparing attack {}, domain {}, distance metric {}, type {} with kwargs: {}'.format(
        attack_name, domain, metric, attack_type, kwargs 
    ))

    if attack_type == 'evasion' and defended_model is None:
        raise ValueError('Evasion attacks require a defended model.')

    if attack_type == 'standard' and defended_model is not None:
        raise ValueError('Passed a defended_model for a standard attack.')

    if domain == 'cifar10':
        num_classes = 10
    else:
        raise NotImplementedError('Unsupported domain "{}".'.format(domain))

    binary_search = 'enable_binary_search' in kwargs and kwargs['enable_binary_search']
        
    kwargs.pop('enable_binary_search', None)

    if binary_search:
        logger.debug('Enabling binary search for {}'.format(attack_name))
        # Remove standard arguments
        kwargs.pop('eps', None)
        if attack_name not in attacks_with_binary_search:
            raise NotImplementedError('Attack {} does not support binary search.'.format(attack_name))
    elif attack_name in attacks_with_binary_search:
        logger.debug('Binary search is supported for {}, but not enabled.'.format(binary_search))

    # Pop binary search arguments
    min_eps = kwargs.pop('min_eps', None)
    max_eps = kwargs.pop('max_eps', None)
    initial_search_steps = kwargs.pop('initial_search_steps', None)
    binary_search_steps = kwargs.pop('binary_search_steps', None)

    if attack_name == 'bim':
        if metric == 'l2':
            if attack_type == 'standard':
                attack = advertorch.attacks.L2BasicIterativeAttack(model)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        elif metric == 'linf':
            if attack_type == 'standard':
                attack = advertorch.attacks.LinfBasicIterativeAttack(model)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    elif attack_name == 'carlini':
        if metric == 'l2':
            if attack_type == 'standard':
                attack = advertorch.attacks.CarliniWagnerL2Attack(model, num_classes, **kwargs)
            elif attack_type == 'evasion':
                # TODO: è la scelta migliore?
                attack = advertorch.attacks.CarliniWagnerL2Attack(defended_model, num_classes + 1, targeted=True, **kwargs)
                attack = evasion_attacks.TopKTargetEvasionAttack(model, attack)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        elif metric == 'linf':
            if attack_type == 'standard':
                attack = standard_attacks.CarliniWagnerLInfAttack(model, num_classes, **kwargs)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    elif attack_name == 'deepfool':
        if metric == 'l2':
            if attack_type == 'standard':
                attack = standard_attacks.L2DeepFoolAttack(model, **kwargs)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        elif metric == 'linf':
            if attack_type == 'standard':
                attack = standard_attacks.LInfDeepFoolAttack(model, **kwargs)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    elif attack_name == 'fast_gradient':
        # FGM is the L2 variant, FGSM is the LInf variant
        if metric == 'l2':
            if attack_type == 'standard':
                attack = advertorch.attacks.FGM(model, **kwargs)
            elif attack_type == 'evasion':
                # TODO: è solo un placeholder
                attack = advertorch.attacks.FGM(defended_model, **kwargs)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        elif metric == 'linf':
            if attack_type == 'standard':
                attack = advertorch.attacks.FGSM(model, **kwargs)
            elif attack_type == 'evasion':
                # TODO: è solo un placeholder
                attack = advertorch.attacks.FGSM(defended_model, **kwargs)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    elif attack_name == 'pgd':
        if metric == 'l2':
            if attack_type == 'standard':
                attack = advertorch.attacks.L2PGDAttack(model, **kwargs)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        elif metric == 'linf':
            if attack_type == 'standard':
                attack = advertorch.attacks.LinfPGDAttack(model, **kwargs)
            else:
                raise NotImplementedError('Unsupported attack "{}" for "{}" of type "{}".'.format(attack_name, metric, attack_type))
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    else:
        raise NotImplementedError('Unsupported attack "{}".'.format(attack_name))


    # If necessary, wrap the attack in a binary search wrapper
    if binary_search:
        if attack_name in ['bim', 'pgd']:
            unsqueeze = False
        elif attack_name == 'fast_gradient':
            unsqueeze = True

        binary_search_kwargs = dict()
        if min_eps is not None:
            binary_search_kwargs['min_eps'] = min_eps
        if max_eps is not None:
            binary_search_kwargs['max_eps'] = max_eps
        if initial_search_steps is not None:
            binary_search_kwargs['initial_search_steps'] = initial_search_steps
        if binary_search_steps is not None:
            binary_search_kwargs['binary_search_steps'] = binary_search_steps

        if attack_type == 'evasion':
            binary_search_kwargs['defended_model'] = defended_model

        attack = custom_attacks.EpsilonBinarySearchAttack(model, p, attack, unsqueeze, **binary_search_kwargs)

    return attack

def get_attack_pool(attack_names, domain, p, attack_type, model, attack_config, defended_model=None):
    attacks = []
    for attack_name in attack_names:
        attacks.append(get_attack(attack_name, domain, p, attack_type, model, attack_config, defended_model=defended_model))

    if len(attacks) == 1:
        return attacks[0]
    else:
        return custom_attacks.AttackPool(model, attacks, p)

def get_detector(attack_name, domain, p, attack_type, model, attack_config, device, substitute_architecture=None, substitute_state_dict_path=None):
    model.to(device)

    if substitute_architecture is not None:
        assert substitute_state_dict_path is not None

    attack = get_attack(attack_name, domain, p, attack_type, model, attack_config, defended_model=None)
    detector = detectors.CounterAttackDetector(attack, model, p)

    if substitute_architecture is not None:
        substitute_detector = get_model(domain, substitute_architecture, substitute_state_dict_path, True, load_weights=True, as_detector=True)

        # The substitute model returns a [batch_size, 1] matrix, while we need a [batch_size] vector
        substitute_detector = torch.nn.Sequential(substitute_detector, torch_utils.Squeeze(1))

        substitute_detector.to(device)

        detector = advertorch.bpda.BPDAWrapper(detector, forwardsub=substitute_detector)

    return detector

def get_detector_pool(attack_names, domain, p, attack_type, model, attack_config, device, substitute_architectures=None, substitute_state_dict_paths=None):
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

        detector = get_detector(attack_names[i], domain, p, attack_type, model, attack_config, device, substitute_architecture=substitute_architecture, substitute_state_dict_path=substitute_state_dict_path)
        detector_pool.append(detector)

    if len(detector_pool) == 1:
        return detector_pool[0]
    else:
        return detectors.DetectorPool(detector_pool, p)

_distance_to_p = {'l2': 2, 'linf' : np.inf}

def validate_lp_distance(ctx, param, value):
    if value == 'l2':
        return 2
    elif value == 'linf':
        return np.inf
    else:
        raise NotImplementedError('Unsupported distance metric "{}."'.format(value))

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