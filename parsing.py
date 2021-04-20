import logging

import advertorch
import advertorch.bpda
import click
import numpy as np
import torch
import torchvision

import attacks
import detectors
import models
import torch_utils
import training
import utils

logger = logging.getLogger(__name__)

domains = ['cifar10', 'mnist']
architectures = ['a', 'b', 'c', 'wong_small', 'wong_large']
attack_types = ['defense', 'evasion', 'standard', 'training']
supported_attacks = ['bim', 'brendel', 'carlini',
                     'deepfool', 'fast_gradient', 'mip', 'pgd', 'uniform']
epsilon_attacks = ['bim', 'fast_gradient', 'pgd', 'uniform']
attacks_with_binary_search = ['bim', 'fast_gradient', 'pgd', 'uniform']
targeted_attacks = ['bim', 'carlini', 'brendel', 'fast_gradient', 'mip', 'pgd']
er_attacks = ['bim', 'carlini', 'pgd', 'uniform']
fb_binary_search_attacks = ['brendel', 'deepfool']
distances = ['l2', 'linf']
misclassification_policies = ['ignore', 'remove', 'use_predicted']
log_levels = ['debug', 'info', 'warning', 'error', 'critical']
_log_level_to_number = {'debug': logging.DEBUG,
                        'info': logging.INFO,
                        'warning': logging.WARNING,
                        'error': logging.ERROR,
                        'critical': logging.CRITICAL}
_distance_to_p = {'l2': 2, 'linf': np.inf}

training_options = [
    click.option('--optimiser', type=click.Choice(['adam', 'sgd']), default='adam', show_default=True,
                 help='The optimiser that will be used for training.'),
    click.option('--learning-rate', type=float, default=1e-3, show_default=True,
                 help='The learning rate for the optimiser.'),
    click.option('--weight-decay', type=float, default=0, show_default=True,
                 help='The weight decay for the optimiser.'),
    click.option('--adam-betas', nargs=2, type=click.Tuple([float, float]), default=(0.9, 0.999), show_default=True,
                 help='The two beta values. Ignored if the optimiser is not "adam".'),
    click.option('--adam-epsilon', type=float, default=1e-8, show_default=True,
                 help='The value of epsilon. Ignored if the optimiser is not "adam".'),
    click.option('--adam-amsgrad', is_flag=True,
                 help='Enables AMSGrad. Ignored if the optimiser is not "adam".'),
    click.option('--sgd-momentum', type=float, default=0, show_default=True,
                 help='The intensity of momentum. Ignored if the optimiser is not "sgd".'),
    click.option('--sgd-dampening', type=float, default=0, show_default=True,
                 help='The intensity of dampening. Ignored if the optimiser is not "sgd".'),
    click.option('--sgd-nesterov', is_flag=True,
                 help='Enables Nesterov Accelerated Gradient. Ignored if the optimiser is not "adam".'),
    click.option('--l1-regularization', type=float, default=0, show_default=True,
                 help='The weight of L1 regularization. 0 disables L1 regularization.'),
    click.option('--validation-dataset', default=None,
                 help='Validation dataset. Mutually exclusive with --validation-split.'),
    click.option('--validation-split', type=float, default=0,
                 help='Uses a portion (0-1) of the train dataset as validation dataset. 0 disables the split. '
                 'Mutually exclusive with --validation-dataset.'),
    click.option('--early-stopping', type=click.IntRange(0, None), default=0, show_default=True,
                 help='The patience of early stopping. 0 disables early stopping. Requires either '
                 '--validation-dataset or --validation-split.'),
    click.option('--early-stopping-delta', type=float, default=0, show_default=True,
                 help='The minimum improvement required to reset early stopping\'s patience.'),
    click.option('--shuffle', type=bool, default=True, show_default=True),
    click.option('--checkpoint-every', type=click.IntRange(1), default=None,
                help='How often the program saves a checkpoint.'),
    click.option('--load-checkpoint', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
                help='If passed, the program will load an existing checkpoint.'),
    click.option('--choose-best', is_flag=True,
                help='If passed, the program will save the state_dict with the best validation loss, otherwise '
                'the state_dict of the last epoch will be saved. Requires either --validation-dataset or '
                '--validation-split.')
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


def set_log_level(log_level):
    logging.getLogger().setLevel(_log_level_to_number[log_level])


def parse_model(domain, architecture, state_dict_path, apply_normalisation, masked_relu, use_grad, load_weights=False, as_detector=False):
    logger.debug('Parsing model with %s-%s, state dict at %s, apply_normalisation=%s, '
                 'masked_relu=%s, load_weights=%s, as_detector=%s.', domain, architecture,
                 state_dict_path, apply_normalisation, masked_relu, load_weights, as_detector)
    if as_detector:
        num_classes = 1
    else:
        num_classes = 10

    pretrained = load_weights and state_dict_path is None

    if pretrained:
        logger.info('No state dict path provided. Using pretrained model.')

    if domain == 'cifar10':
        model = models.cifar10(architecture, masked_relu,
                               pretrained=pretrained, num_classes=num_classes)
    elif domain == 'mnist':
        model = models.mnist(architecture, masked_relu,
                             pretrained=pretrained, num_classes=num_classes)
    else:
        raise NotImplementedError(f'Unsupported domain {domain}.')

    if apply_normalisation:
        # CIFAR10:
        # mean = np.array([0.4914, 0.4822, 0.4465])
        # std = np.array([0.2023, 0.1994, 0.2010])

        if domain == 'cifar10':
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            num_channels = 3
        elif domain == 'mnist':
            mean = np.array([0.1307])
            std = np.array([0.3081])
            num_channels = 1
        else:
            raise NotImplementedError(
                f'Unsupported normalisation for domain {domain}.')

        logger.debug('Added normalisation (mean=%s, std=%s).', mean, std)

        normalisation = torch_utils.Normalisation(
            mean, std, num_channels=num_channels)
        model = torch.nn.Sequential(normalisation, model)

    # Nota: Questo fa sì che i modelli vengano salvati come modello con normalisation
    if load_weights and state_dict_path is not None:
        logger.info('Loading weights from %s.', state_dict_path)
        model.load_state_dict(torch.load(state_dict_path))

    if not use_grad:
        logger.debug('Permanently disabling gradients for model.')
        _ = torch_utils.disable_model_gradients(model)

    return model


def parse_dataset(domain, dataset, allow_standard=True, dataset_edges=None, extra_transforms=None):
    if extra_transforms is None:
        extra_transforms = []

    logger.debug('Parsing dataset %s-%s (edges=%s) with %s extra transforms.',
                 domain, dataset, dataset_edges, len(extra_transforms))
    matched_dataset = None
    tensor_transform = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose(
        extra_transforms + [tensor_transform])

    if allow_standard:
        if domain == 'cifar10':
            if dataset == 'std:train':
                matched_dataset = torchvision.datasets.CIFAR10(
                    './data/cifar10', train=True, download=True, transform=transform)
            elif dataset == 'std:test':
                matched_dataset = torchvision.datasets.CIFAR10(
                    './data/cifar10', train=False, download=True, transform=transform)
        elif domain == 'mnist':
            if dataset == 'std:train':
                matched_dataset = torchvision.datasets.MNIST(
                    './data/mnist', train=True, download=True, transform=transform)
            elif dataset == 'std:test':
                matched_dataset = torchvision.datasets.MNIST(
                    './data/mnist', train=False, download=True, transform=transform)

    if matched_dataset is None:
        logger.debug('No standard dataset found, interpreting it as a file path.')

        try:
            matched_dataset = utils.load_zip(dataset)
        except:
            raise RuntimeError(
                f'Could not find a standard dataset or a dataset file "{dataset}".')

    if dataset_edges is not None:
        start, stop = dataset_edges
        matched_dataset = training.StartStopDataset(
            matched_dataset, start=start, stop=stop)

    return matched_dataset


def parse_optimiser(optimiser_name, learnable_parameters, options):
    logger.debug('Parsing optimiser %s with options %s', optimiser_name, options)
    if optimiser_name == 'adam':
        optimiser = torch.optim.Adam(
            learnable_parameters, lr=options['learning_rate'],
            betas=options['adam_betas'], weight_decay=options['weight_decay'],
            eps=options['adam_epsilon'], amsgrad=options['adam_amsgrad'])
    elif optimiser_name == 'sgd':
        optimiser = torch.optim.SGD(
            learnable_parameters, lr=options['learning_rate'], momentum=options['sgd_momentum'],
            dampening=options['sgd_dampening'], weight_decay=options['weight_decay'], nesterov=options['sgd_nesterov'])
    else:
        raise ValueError(f'Unsupported optimiser "{optimiser_name}".')

    return optimiser

# Targeted FGSM is introduced in
# http://bengio.abracadoudou.com/publications/pdf/kurakin_2017_iclr_physical.pdf

def parse_attack(attack_name, domain, p, attack_type, model, attack_config, device, defended_model=None, seed=None):
    logger.debug('Parsing %s attack %s-%s (using defended: %s).', attack_type,
                 attack_name, p, defended_model is not None)

    # Convert the float value to its standard name
    if p == 2:
        metric = 'l2'
    elif np.isposinf(p):
        metric = 'linf'
    else:
        raise NotImplementedError(f'Unsupported metric "l{p}"')

    if attack_type not in attack_types:
        raise NotImplementedError(f'Unsupported attack type {attack_type}.')

    kwargs = attack_config.get_arguments(
        attack_name, domain, metric, attack_type)

    logger.debug('Loaded attack kwargs: %s.', kwargs)

    if attack_name == 'uniform' and metric != 'linf':
        logger.warning('UniformAttack is designed for the LInf metric. Are you sure that you '
                       'want to use %s?', metric)

    if attack_type == 'evasion' and defended_model is None:
        raise ValueError('Evasion attacks require a defended model.')

    binary_search = 'enable_binary_search' in kwargs and kwargs['enable_binary_search']
    return_best = kwargs.pop('return_best', False)

    if attack_type != 'evasion' and defended_model is not None:
        raise ValueError(
            'Passed a defended_model for a non-evasion attack.')

    if domain in ['cifar10', 'mnist']:
        num_classes = 10
    else:
        raise NotImplementedError(f'Unsupported domain "{domain}".')

    evade_detector = (attack_type == 'evasion')

    if evade_detector:
        num_classes += 1

    kwargs.pop('enable_binary_search', None)

    if binary_search:
        logger.debug('Enabling binary search for %s.', attack_name)
        # Remove standard arguments
        kwargs.pop('eps', None)
        if attack_name not in attacks_with_binary_search:
            raise NotImplementedError(
                f'Attack {attack_name} does not support binary search.')
    elif attack_name in attacks_with_binary_search:
        logger.warning(
            'Binary search for attack %s is disabled in the configuration file, despite being supported.', attack_name)

    # Pop binary search arguments
    min_eps = kwargs.pop('min_eps', None)
    max_eps = kwargs.pop('max_eps', None)
    eps_initial_search_steps = kwargs.pop('eps_initial_search_steps', None)
    eps_binary_search_steps = kwargs.pop('eps_binary_search_steps', None)

    # Pop epsilon attack arguments
    force_eps = kwargs.pop('force_eps', None)

    if evade_detector:
        target_model = defended_model
    else:
        target_model = model

    # TODO: Check compatibility between evasion and return_best
    if return_best and not (attack_name == 'carlini' and np.isposinf(p)):
        logger.debug('Wrapping in BestSampleWrapper.')
        target_model = attacks.BestSampleWrapper(target_model)

    if attack_name == 'bim':
        if metric == 'l2':
            attack = advertorch.attacks.L2BasicIterativeAttack(
                target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.LinfBasicIterativeAttack(
                target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError(
                f'Unsupported attack "{attack_name}" for "{metric}".')
    elif attack_name == 'brendel':
        attack = attacks.BrendelBethgeAttack(target_model, p, **kwargs)
    elif attack_name == 'carlini':
        if metric == 'l2':
            attack = advertorch.attacks.CarliniWagnerL2Attack(
                target_model, num_classes, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            cuda_optimized = device == 'cuda'
            attack = attacks.get_carlini_linf_attack(target_model, num_classes,
                targeted=evade_detector, return_best=return_best, cuda_optimized=cuda_optimized, **kwargs)
        else:
            raise NotImplementedError(
                f'Unsupported attack "{attack_name}" for "{metric}".')
    elif attack_name == 'deepfool':
        attack = attacks.DeepFoolAttack(target_model, p, **kwargs)
    elif attack_name == 'fast_gradient':
        # FGM is the L2 variant, FGSM is the Linf variant
        if metric == 'l2':
            attack = advertorch.attacks.FGM(
                target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.FGSM(
                target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError(
                f'Unsupported attack "{attack_name}" for "{metric}".')
    elif attack_name == 'mip':
        if attack_type == 'evasion':
            raise NotImplementedError('MIP does not support evasion.')
        attack = attacks.MIPAttack(
            target_model, p, targeted=evade_detector, seed=seed, **kwargs)
    elif attack_name == 'pgd':
        if metric == 'l2':
            attack = advertorch.attacks.L2PGDAttack(
                target_model, targeted=evade_detector, **kwargs)
        elif metric == 'linf':
            attack = advertorch.attacks.LinfPGDAttack(
                target_model, targeted=evade_detector, **kwargs)
        else:
            raise NotImplementedError(
                f'Unsupported attack "{attack_name}" for "{metric}".')
    elif attack_name == 'uniform':
        attack = attacks.UniformNoiseAttack(
            target_model, p, targeted=evade_detector, **kwargs)
    else:
        raise NotImplementedError(f'Unsupported attack "{attack_name}".')

    # Add support for epsilon
    if attack_name in epsilon_attacks:
        if attack_name in ['bim', 'pgd']:
            unsqueeze = False
        elif attack_name in ['fast_gradient', 'uniform']:
            unsqueeze = True
        else:
            raise ValueError(
                f'Unsupported epsilon attack "{attack_name}"')

        epsilon_attack_kwargs = {}

        if force_eps is not None:
            epsilon_attack_kwargs['force_eps'] = force_eps

        attack = attacks.EpsilonAttack(attack, unsqueeze, **epsilon_attack_kwargs)

    # If necessary, wrap the attack in a binary search wrapper
    if binary_search:
        logger.debug('Adding binary search.')
        binary_search_kwargs = dict()
        if min_eps is not None:
            binary_search_kwargs['min_eps'] = min_eps
        if max_eps is not None:
            binary_search_kwargs['max_eps'] = max_eps
        if eps_initial_search_steps is not None:
            binary_search_kwargs['eps_initial_search_steps'] = eps_initial_search_steps
        if eps_binary_search_steps is not None:
            binary_search_kwargs['eps_binary_search_steps'] = eps_binary_search_steps

        attack = attacks.EpsilonBinarySearchAttack(
            attack, p, targeted=evade_detector, **binary_search_kwargs)

    # Complete the best sample wrapping
    # Carlini Linf does not support BestSample
    if return_best and not (attack_name == 'carlini' and np.isposinf(p)):
        logger.debug('Finalizing best sample wrapping.')
        suppress_warning = attack_name in fb_binary_search_attacks
        attack = attacks.BestSampleAttack(
            target_model, attack, p, targeted=evade_detector, suppress_warning=suppress_warning)

    # Convert targeted evasion attacks into untargeted ones
    if evade_detector and (attack_name in targeted_attacks):
        logger.debug('Converting targeted to untargeted attack.')
        attack = attacks.KBestTargetEvasionAttack(target_model, attack)

    return attack


def parse_attack_pool(attack_names, domain, p, attack_type, model, attack_config, device, defended_model=None, seed=None):
    logger.debug('Parsing %s attack pool %s for %s (using defended: %s).', attack_type,
                 attack_names, p, defended_model is not None)
    evade_detector = (attack_type == 'evasion')

    if evade_detector:
        target_model = defended_model
    else:
        target_model = model

    attack_pool = []
    for attack_name in attack_names:
        attack_pool.append(parse_attack(attack_name, domain, p, attack_type,
                                      model, attack_config, device, defended_model=defended_model, seed=seed))

    if len(attack_pool) == 1:
        return attack_pool[0]
    else:
        return attacks.AttackPool(target_model, evade_detector, attack_pool, p)


def parse_detector(attack_name, domain, p, attack_type, model, attack_config, device, use_substitute=False, substitute_state_dict_path=None):
    logger.debug(
        f'Preparing detector for "{p}" of type "{attack_type}" with attack "{attack_name}".')

    model.to(device)

    if use_substitute:
        assert substitute_state_dict_path is not None

    if attack_type != 'defense':
        logger.warning(
            f'You are using an attack of type "{attack_type}" for a detector. Is this intentional?')

    attack = parse_attack(attack_name, domain, p, attack_type,
                        model, attack_config, device, defended_model=None)
    assert attack.predict == model
    detector = detectors.CounterAttackDetector(attack, model, p)

    if use_substitute:
        # TODO: è corretto che use_grad=False?
        # TODO: Valore di seed
        seed = None
        substitute_detector = parse_model(
            domain, substitute_state_dict_path, True, False, load_weights=True, as_detector=True, seed=seed)

        # The substitute model returns a [batch_size, 1] matrix, while we need a [batch_size] vector
        substitute_detector = torch.nn.Sequential(
            substitute_detector, torch_utils.Squeeze(1))

        substitute_detector.to(device)

        detector = advertorch.bpda.BPDAWrapper(
            detector, forwardsub=substitute_detector)

    return detector


def parse_detector_pool(attack_names, domain, p, attack_type, model, attack_config, device, use_substitute=False, substitute_state_dict_paths=None):
    if use_substitute:
        assert len(substitute_state_dict_paths) == len(attack_names)

    logger.debug(
        f'Preparing detector pool for "{p}" of type "{attack_type}" containing {attack_names}.')

    detector_pool = []

    for i in range(len(attack_names)):
        substitute_state_dict_path = None
        if use_substitute:
            substitute_state_dict_path = substitute_state_dict_paths[i]

        detector = parse_detector(attack_names[i], domain, p, attack_type, model, attack_config, device,
                                use_substitute=use_substitute, substitute_state_dict_path=substitute_state_dict_path,)
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
        raise NotImplementedError(f'Unsupported distance metric "{value}".')


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
                parameter_list = [parameter.strip()
                                  for parameter in value.split(',')]
            except:
                raise click.BadParameter(
                    'Parameter must be in format "[value_1, value_2, ...]" (with quotes, if there are spaces).')
        else:
            parameter_list = [value]

        if self.allowed_values is not None:
            for parameter in parameter_list:
                if parameter not in self.allowed_values:
                    raise click.BadParameter(
                        f'Unrecognised value "{parameter}".')

        if self.cast_to is not None:
            parameter_list = [self.cast_to(parameter)
                              for parameter in parameter_list]

        return parameter_list
