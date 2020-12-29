import logging

import advertorch
import advertorch.bpda
import click
import attacks
import attacks
import numpy as np
import torch
import torchvision

import attacks
import cifar10_models
import detectors
import models
import torch_utils
import utils

logger = logging.getLogger(__name__)

domains = ['cifar10', 'mnist', 'svhn']
architectures = ['a', 'b', 'c', 'wong_small', 'wong_large', 'small', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
supported_attacks = ['bim', 'carlini', 'brendel', 'deepfool', 'fast_gradient', 'mip', 'pgd', 'uniform']
attacks_with_binary_search = ['bim', 'fast_gradient', 'pgd', 'uniform']
targeted_attacks = ['bim', 'carlini', 'fast_gradient', 'mip', 'pgd']
er_attacks = ['bim', 'carlini', 'pgd', 'uniform']
foolbox_attacks = ['brendel', 'deepfool']
distances = ['l2', 'linf']
misclassification_policies = ['ignore', 'remove', 'use_predicted']
log_levels = ['debug', 'info', 'warning', 'error', 'critical']
_log_level_to_number = {'debug' : logging.DEBUG,
                        'info' : logging.INFO,
                        'warning' : logging.WARNING,
                        'error' : logging.ERROR,
                        'critical' : logging.CRITICAL}
_distance_to_p = {'l2': 2, 'linf' : np.inf}

training_options = [
    click.option('--optimiser', type=click.Choice(['adam', 'sgd']), default='adam', show_default=True,
        help='The optimiser that will be used for training.'),
    click.option('--learning-rate', type=float, default=1e-3, show_default=True,
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
    click.option('--l1-regularization', type=float, default=0, show_default=True,
        help='The weight of L1 regularization. 0 disables L1 regularization'),
    click.option('--validation-dataset', default=None),
    click.option('--validation-split', type=float, default=0,
        help='Uses a portion (0-1) of the train dataset as validation dataset. 0 disables the split.'),
    click.option('--early-stopping', type=click.IntRange(0, None), default=0, show_default=True,
        help='The patience of early stopping. 0 disables early stopping.'),
    click.option('--early-stopping-delta', type=float, default=0, show_default=True,
        help='The minimum improvement required to reset early stopping\'s patience.'),
    click.option('--shuffle', type=bool, default=True)
]

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options

def set_log_level(log_level):
    logging.getLogger().setLevel(_log_level_to_number[log_level])

def get_model(domain, architecture, state_dict_path, apply_normalisation, masked_relu, load_weights=False, as_detector=False):
    if as_detector:
        num_classes = 1
    else:
        num_classes = 10
    
    pretrained = load_weights and state_dict_path is None

    if pretrained:
        logger.info('No state dict path provided. Using pretrained model.')

    if domain == 'cifar10':
        model = models.cifar10(architecture, masked_relu, pretrained=pretrained, num_classes=num_classes)
    elif domain == 'svhn':
        model = models.svhn(architecture, masked_relu, pretrained=pretrained, num_classes=num_classes)
    elif domain == 'mnist':
        model = models.mnist(architecture, masked_relu, pretrained=pretrained, num_classes=num_classes)
    else:
        raise NotImplementedError('Unsupported domain {}.'.format(domain))

    if apply_normalisation:
        # CIFAR10:
        # mean = np.array([0.4914, 0.4822, 0.4465])
        # std = np.array([0.2023, 0.1994, 0.2010])
        # SVHN
        # mean = np.array([0.4377, 0.4438, 0.4728])
        # std = np.array([0.1201, 0.1231, 0.1052])

        # The pretrained CIFAR10 and SVHN models
        # use the standard 0.5 normalisations
        if domain in ['cifar10', 'svhn']:
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])
            num_channels = 3
        elif domain == 'mnist':
            mean = np.array([0.1307])
            std = np.array([0.3081])
            num_channels = 1
        else:
            raise NotImplementedError('Unsupported normalisation for domain {}.'.format(domain))

        normalisation = torch_utils.Normalisation(mean, std, num_channels=num_channels)
        model = torch.nn.Sequential(normalisation, model)

    # Nota: Questo fa sì che i modelli vengano salvati come modello con normalisation
    if load_weights and state_dict_path is not None:
        logger.info('Loading weights from {}'.format(state_dict_path))
        model.load_state_dict(torch.load(state_dict_path))

    return model

def get_dataset(domain, dataset, allow_standard=True, start=None, stop=None, extra_transforms=[]):
    matched_dataset = None
    tensor_transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose(extra_transforms + [tensor_transform])
    
    if allow_standard:
        if domain == 'cifar10':
            if dataset == 'std:train':
                matched_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform)
            elif dataset == 'std:test':
                matched_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transform)
        elif domain == 'svhn':
            if dataset == 'std:train':
                matched_dataset = torchvision.datasets.SVHN('./data/svhn', split='train', download=True, transform=transform)
            elif dataset == 'std:extra':
                matched_dataset = torchvision.datasets.SVHN('./data/svhn', split='extra', download=True, transform=transform)
            elif dataset == 'std:test':
                matched_dataset = torchvision.datasets.SVHN('./data/svhn', split='test', download=True, transform=transform)
        elif domain == 'mnist':
            if dataset == 'std:train':
                matched_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
            elif dataset == 'std:test':
                matched_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)

    if matched_dataset is None:
        # No matches found, try to read it as a file path
        try:
            matched_dataset = utils.load_zip(dataset)
        except:
            raise RuntimeError('Could not find a standard dataset or a dataset file "{}".'.format(dataset))

    if (start is not None and start != 0) or stop is not None:
        matched_dataset = torch_utils.StartStopDataset(matched_dataset, start=start, stop=stop)

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

# Targeted FGSM is introduced in
# http://bengio.abracadoudou.com/publications/pdf/kurakin_2017_iclr_physical.pdf

def get_attack(attack_name, domain, p, attack_type, model, attack_config, defended_model=None, early_rejection_threshold=None):
    # Convert the float value to its standard name
    if p == 2:
        metric = 'l2'
    elif np.isposinf(p):
        metric = 'linf'
    else:
        raise NotImplementedError('Unsupported metric "l{}"'.format(p))

    kwargs = attack_config.get_arguments(attack_name, domain, metric, attack_type)

    logger.debug('Preparing attack "{}", domain "{}", distance metric "{}", type "{}" with kwargs: {}'.format(
        attack_name, domain, metric, attack_type, kwargs 
    ))

    if attack_type == 'evasion' and defended_model is None:
        raise ValueError('Evasion attacks require a defended model.')

    early_rejection = kwargs.pop('early_rejection', None)
    binary_search = 'enable_binary_search' in kwargs and kwargs['enable_binary_search']
    return_best = kwargs.pop('return_best', False)

    if early_rejection:
        logger.debug('Enabled early rejection for "{}" ({}).'.format(attack_name, early_rejection_threshold))
        if attack_name not in er_attacks:
            if binary_search:
                logger.warning('Attack "{}" does not support early rejection, but binary search is enabled. '
                'Early rejection will be only applied to binary search.'.format(attack_name))
            else:
                raise ValueError('Attack "{}" does not support early rejection.'.format(attack_name))

        if early_rejection_threshold is None:
            raise ValueError('Passed None early_rejection_threshold for an attack with early_rejection enabled.')

        if early_rejection_threshold <= 0:
            logger.warning('Attack "{}" is using a nonpositive early_rejection_threshold. This means that early rejection is '
                            'basically disabled.'.format(attack_name))

        if attack_type != 'defense':
            logger.warning('Attack "{}" is using early rejection despite being of type "{}". Is this intentional?'.format(attack_name, attack_type))

        if attack_name in er_attacks:
            kwargs['early_rejection_threshold'] = early_rejection_threshold
    elif attack_type == 'defense' and attack_name in er_attacks and early_rejection_threshold is not None:
        logger.warning('Early rejection for attack "{}" is disabled in the configuration file, despite being supported.'.format(attack_name))


    if (attack_type == 'standard' or attack_type == 'defense') and defended_model is not None:
        raise ValueError('Passed a defended_model for a standard/defense attack.')

    if domain in ['cifar10', 'mnist', 'svhn']:
        num_classes = 10
    else:
        raise NotImplementedError('Unsupported domain "{}".'.format(domain))

    evade_detector = (attack_type == 'evasion')

    if evade_detector:
        num_classes += 1

    kwargs.pop('enable_binary_search', None)

    if binary_search:
        logger.debug('Enabling binary search for "{}".'.format(attack_name))
        # Remove standard arguments
        kwargs.pop('eps', None)
        if attack_name not in attacks_with_binary_search:
            raise NotImplementedError('Attack {} does not support binary search.'.format(attack_name))
    elif attack_name in attacks_with_binary_search:
        logger.warning('Binary search for attack "{}" is disabled in the configuration file, despite being supported.'.format(attack_name))

    # Pop binary search arguments
    min_eps = kwargs.pop('min_eps', None)
    max_eps = kwargs.pop('max_eps', None)
    eps_initial_search_steps = kwargs.pop('eps_initial_search_steps', None)
    eps_binary_search_steps = kwargs.pop('eps_binary_search_steps', None)

    if evade_detector:
        target_model = defended_model
    else:
        target_model = model

    # TODO: Check compatibility between evasion and return_best
    if return_best:
        target_model = attacks.BestSampleWrapper(target_model)

    if attack_name == 'bim':
        if metric == 'l2':
            attack = advertorch.attacks.L2BasicIterativeAttack(target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.LinfBasicIterativeAttack(target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    elif attack_name == 'brendel':
        attack = attacks.BrendelBethgeAttack(target_model, p, **kwargs)
    elif attack_name == 'carlini':
        if metric == 'l2':
            attack = advertorch.attacks.CarliniWagnerL2Attack(target_model, num_classes, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.CarliniWagnerLinfAttack(target_model, num_classes, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    elif attack_name == 'deepfool':
        attack = attacks.DeepFoolAttack(target_model, p, **kwargs)
    elif attack_name == 'fast_gradient':
        # FGM is the L2 variant, FGSM is the LInf variant
        if metric == 'l2':
            attack = advertorch.attacks.FGM(target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.FGSM(target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    elif attack_name == 'mip':
        if attack_type == 'evasion':
            raise NotImplementedError('MIP does not support evasion.')
        attack = attacks.MIPAttack(target_model, p, targeted=evade_detector, **kwargs)
    elif attack_name == 'pgd':
        if metric == 'l2':
            attack = advertorch.attacks.L2PGDAttack(target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.LinfPGDAttack(target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError('Unsupported attack "{}" for "{}".'.format(attack_name, metric))
    elif attack_name == 'uniform':
         attack = attacks.UniformNoiseAttack(target_model, p, targeted=evade_detector, **kwargs)
    else:
        raise NotImplementedError('Unsupported attack "{}".'.format(attack_name))

    if 'stochastic_consistency' in kwargs and kwargs['stochastic_consistency']:
        logger.warn('Stochastic consistency is deprecated.')

    # If necessary, wrap the attack in a binary search wrapper
    if binary_search:
        if attack_name in ['bim', 'pgd']:
            unsqueeze = False
        elif attack_name in ['fast_gradient', 'uniform']:
            unsqueeze = True
        else:
            raise ValueError('Unsupported binary search attack "{}"'.format(attack_name))

        binary_search_kwargs = dict()
        if min_eps is not None:
            binary_search_kwargs['min_eps'] = min_eps
        if max_eps is not None:
            binary_search_kwargs['max_eps'] = max_eps
        if eps_initial_search_steps is not None:
            binary_search_kwargs['eps_initial_search_steps'] = eps_initial_search_steps
        if eps_binary_search_steps is not None:
            binary_search_kwargs['eps_binary_search_steps'] = eps_binary_search_steps

        if early_rejection:
            binary_search_kwargs['early_rejection_threshold'] = early_rejection_threshold

        attack = attacks.EpsilonBinarySearchAttack(target_model, evade_detector, p, attack, unsqueeze, targeted=evade_detector, **binary_search_kwargs)

    if return_best:
        suppress_warning = attack_name in foolbox_attacks
        attack = attacks.BestSampleAttack(target_model, attack, p, evade_detector, suppress_warning=suppress_warning)

    # Convert targeted evasion attacks into untargeted ones
    if evade_detector and (attack_name in targeted_attacks):
        attack = attacks.KBestTargetEvasionAttack(model, attack)

    return attack

def get_attack_pool(attack_names, domain, p, attack_type, model, attack_config, defended_model=None, early_rejection_threshold=None):
    evade_detector = (attack_type == 'evasion')

    if evade_detector:
        target_model = defended_model
    else:
        target_model = model

    logger.debug('Preparing attack pool for "{}" of type "{}" containing {} (with defended model: {}) '
    'and early rejection threshold {}.'.format(p, attack_type, attack_names, defended_model is not None, early_rejection_threshold))

    attack_pool = []
    for attack_name in attack_names:
        attack_pool.append(get_attack(attack_name, domain, p, attack_type, model, attack_config, defended_model=defended_model, early_rejection_threshold=early_rejection_threshold))

    if len(attack_pool) == 1:
        return attack_pool[0]
    else:
        return attacks.AttackPool(target_model, evade_detector, attack_pool, p)

def get_detector(attack_name, domain, p, attack_type, model, attack_config, device, use_substitute=False, substitute_state_dict_path=None,
                early_rejection_threshold=None):
    logger.debug('Preparing detector for "{}" of type "{}" with attack "{}" '
    'and early rejection threshold {}.'.format(p, attack_type, attack_name, early_rejection_threshold))

    model.to(device)

    if use_substitute:
        assert substitute_state_dict_path is not None

    if attack_type != 'defense':
        logger.warning('You are using an attack of type "{}" for a detector. Is this intentional?'.format(attack_type))

    attack = get_attack(attack_name, domain, p, attack_type, model, attack_config, defended_model=None, early_rejection_threshold=early_rejection_threshold)
    assert attack.predict == model
    detector = detectors.CounterAttackDetector(attack, model, p)

    if use_substitute:
        substitute_detector = get_model(domain, substitute_state_dict_path, True, load_weights=True, as_detector=True)

        # The substitute model returns a [batch_size, 1] matrix, while we need a [batch_size] vector
        substitute_detector = torch.nn.Sequential(substitute_detector, torch_utils.Squeeze(1))

        substitute_detector.to(device)

        detector = advertorch.bpda.BPDAWrapper(detector, forwardsub=substitute_detector)

    return detector

def get_detector_pool(attack_names, domain, p, attack_type, model, attack_config, device, use_substitute=False, substitute_state_dict_paths=None, early_rejection_threshold=None):
    if use_substitute:
        assert len(substitute_state_dict_paths) == len(attack_names)

    logger.debug('Preparing detector pool for "{}" of type "{}" containing {} '
    'and early rejection threshold {}.'.format(p, attack_type, attack_names, early_rejection_threshold))
    
    detector_pool = []

    for i in range(len(attack_names)):
        substitute_state_dict_path = None
        if use_substitute:
            substitute_state_dict_path = substitute_state_dict_paths[i]

        detector = get_detector(attack_names[i], domain, p, attack_type, model, attack_config, device, use_substitute=use_substitute, substitute_state_dict_path=substitute_state_dict_path,
        early_rejection_threshold=early_rejection_threshold)
        detector_pool.append(detector)

    if len(detector_pool) == 1:
        return detector_pool[0]
    else:
        return detectors.DetectorPool(detector_pool, p)

def validate_lp_distance(ctx, param, value):
    if value == 'l2':
        return 2
    elif value == 'linf':
        return np.inf
    elif value is None:
        return None
    else:
        raise NotImplementedError('Unsupported distance metric "{}."'.format(value))

class ParameterList:
    def __init__(self, allowed_values=None, cast_to=None):
        self.allowed_values = allowed_values
        self.cast_to = cast_to

    def __call__(self, ctx, param, value):
        if value is None:
            return []

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