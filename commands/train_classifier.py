import logging
import pathlib

import click
import torch
import torchvision

import utils
import parsing
import training

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('epochs', type=click.IntRange(1, None))
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--masked-relu', is_flag=True,
              help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.option('--batch-size', type=click.IntRange(1, None), default=50, show_default=True,
              help='The batch size of the dataset.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--cpu-threads', type=click.IntRange(1, None, False), default=None,
              help='The number of PyTorch CPU threads. If unspecified, the default '
              'number is used (usually the number of cores).')
@parsing.add_options(parsing.training_options)
@click.option('--rs-regularization', type=float, default=0, show_default=True,
              help='The weight of RS regularization. 0 disables RS regularization. If enabled, requires specifying --rs-eps.')
@click.option('--rs-eps', type=float, default=None,
              help='The epsilon of RS regularization. Ignored if RS regularization is disabled. '
              'Required if RS regularization is enabled.')
@click.option('--rs-minibatch', type=click.IntRange(1, None), default=None,
              help='If passed, the batch will be split into minibatches of this size before being used to '
              'compute the RS loss. This helps to reduce GPU memory usage. Ignored if RS regularization is disabled.')
@click.option('--rs-start-epoch', type=click.IntRange(1, None), default=1, show_default=True,
              help='The first epoch (1-indexed) where RS loss is activated. 1 makes RS loss always active. '
                   'Ignored if RS regularization is disabled.')
@click.option('--flip', is_flag=True,
              help='Enables random horizontal flipping.')
@click.option('--rotation', type=float, default=0, show_default=True,
              help='Random rotation (in degrees) in range (-value, +value).')
@click.option('--translation', type=float, default=0, show_default=True,
              help='Random horizontal and vertical translation in range (-value * image_size, +value * image_size).')
@click.option('--adversarial-training', callback=parsing.ParameterList(parsing.epsilon_attacks), default=None,
              help='The adversarial attack that will be used to compute the adversarials. '
              'If unspecified, disables adversarial training. Requires specifying --adversarial-ratio, --adversarial-p '
              'and --adversarial-eps.')
@click.option('--adversarial-ratio', type=float, default=None,
              help='The ratio of samples that are replaced with adversarials. '
              'Ignored if adversarial training is disabled. '
              'Required if adversarial training is enabled.')
@click.option('--adversarial-p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance, default=None,
              help='The Lp norm used for adversarial training. '
              'Ignored if adversarial training is disabled. '
              'Required if adversarial training is enabled.')
@click.option('--adversarial-eps', type=float, default=None,
              help='The maximum perturbation of an adversarial. '
              'Ignored if adversarial training is disabled. '
              'Required if adversarial training is enabled.')
@click.option('--adversarial-eps-growth-epoch', type=click.IntRange(0, None), default=0,
              help='If bigger than 0, the adversarial epsilon will grow linearly from --adversarial-eps-growth-start '
              'to --adversarial-eps, which will be reached at the specified epoch (1-indexed). '
              'Requires specifying --adversarial-eps-growth-start. Ignored if adversarial training is disabled.')
@click.option('--adversarial-eps-growth-start', type=float, default=None,
              help='The initial value of eps. Ignored if --adversarial-eps-growth-epoch is unspecified. '
              'Required if --adversarial-eps-growth-epoch is specified.')
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='default_attack_configuration.cfg', show_default=True,
              help='The path to the attack configuration file for adversarial training. Ignored if adversarial training is disabled.')
@click.option('--seed', type=int, default=None,
              help='The seed for random generation. If unspecified, the current time is used as seed.')
@click.option('--deterministic', is_flag=True,
              help='If passed, all computations except random number generation are deterministic (but slower).')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
def train_classifier(**kwargs):
    parsing.set_log_level(kwargs['log_level'])
    logger.debug('Running train-classifier command with kwargs %s', kwargs)

    if kwargs['deterministic']:
        if kwargs['seed'] is None:
            logger.warning('Determinism is enabled, but no seed has been provided.')

        utils.enable_determinism()

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])

    if kwargs['seed'] is not None:
        utils.set_seed(kwargs['seed'])

    model = parsing.parse_model(kwargs['domain'], kwargs['architecture'],
                              None, False, kwargs['masked_relu'], True, load_weights=False)
    model.train()

    extra_transforms = []

    if kwargs['flip']:
        extra_transforms.append(torchvision.transforms.RandomHorizontalFlip())

    if kwargs['rotation'] != 0 or kwargs['translation'] != 0:
        if kwargs['translation'] < 0 or kwargs['translation'] > 1:
            logger.warning('The suggested range for --translation is [0, 1].')

        if kwargs['rotation'] < 0 or kwargs['rotation'] > 180:
            logger.warning('The suggested range for --rotation is [0, 180].')

        translation = (kwargs['translation'], kwargs['translation']
                       ) if kwargs['translation'] != 0 else None
        extra_transforms.append(torchvision.transforms.RandomAffine(
            kwargs['rotation'], translation))

    train_dataset = parsing.parse_dataset(
        kwargs['domain'], kwargs['dataset'], extra_transforms=extra_transforms)

    # Validation
    val_dataset = None

    if kwargs['validation_dataset'] is not None and kwargs['validation_split'] != 0:
        raise click.BadOptionUsage(
            '--validation_split', '--validation_dataset and validation_split are mutually exclusive.')

    if kwargs['validation_split'] != 0:
        logger.debug('Performing a validation split.')
        train_dataset, val_dataset = training.split_dataset(
            train_dataset, kwargs['validation_split'], shuffle=True)
    elif kwargs['validation_dataset'] is not None:
        logger.debug('Loading an existing validation dataset.')
        val_dataset = parsing.parse_dataset(
            kwargs['domain'], kwargs['validation_dataset'], allow_standard=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, kwargs['batch_size'], shuffle=kwargs['shuffle'])
    if val_dataset is None:
        val_dataloader = None
    else:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, kwargs['batch_size'], shuffle=False)

    # Early stopping
    early_stopping = None
    if kwargs['early_stopping'] > 0:
        if kwargs['choose_best'] and kwargs['early_stopping_delta'] != 0:
            logger.warning('Received --choose-best and --early-stopping with delta != 0. '
                           'Remember that with delta != 0, --choose-best and --early-stopping '
                           'track differently the best loss and state_dict.')
        logger.debug('Adding early stopping.')
        early_stopping = training.EarlyStopping(
            kwargs['early_stopping'], delta=kwargs['early_stopping_delta'])

    # Adversarial training
    if kwargs['adversarial_training'] == []:
        adversarial_attack = None

        if kwargs['adversarial_ratio'] is not None:
            logger.warning('Received --adversarial-ratio without --adversarial-training.')
        if kwargs['adversarial_p'] is not None:
            logger.warning('Received --adversarial-p without --adversarial-training.')
        if kwargs['adversarial_eps'] is not None:
            logger.warning('Received --adversarial-eps without --adversarial-training.')
        if kwargs['adversarial_eps_growth_epoch'] != 0:
            logger.warning('Received --adversarial-eps-growth-epoch without --adversarial-training.')
        if kwargs['adversarial_eps_growth_start'] is not None:
            logger.warning('Received --adversarial-eps-growth-start without --adversarial-training.')
    else:
        logger.debug('Enabling adversarial training.')

        if kwargs['adversarial_ratio'] is None:
            raise click.BadOptionUsage(
                '--adversarial-ratio', 'Please specify the ratio for adversarial training with --adversarial-ratio.')

        if kwargs['adversarial_ratio'] <= 0 or kwargs['adversarial_ratio'] > 1:
            raise click.BadOptionUsage(
                '--adversarial-ratio', '--adversarial-ratio must be between 0 (exclusive) and 1 (inclusive).')

        if kwargs['adversarial_p'] is None:
            raise click.BadOptionUsage(
                '--adversarial-p', 'Please specify the Lp norm for adversarial training with --adversarial-p.')

        if kwargs['adversarial_eps'] is None:
            raise click.BadOptionUsage(
                '--adversarial-eps', 'Please specify the maximum perturbarion norm for adversarial training with --adversarial-eps (inf is also allowed).')

        if kwargs['adversarial_eps_growth_epoch'] > 0:
            if kwargs['adversarial_eps_growth_start'] is None:
                raise click.BadOptionUsage(
                    '--adversarial-eps-growth-start', 'Please specify the initial value for adversarial epsilon growth with --adversarial-eps-growth-start '
                    '(0 is also allowed).')

            if kwargs['early_stopping'] > 0:
                logger.warning('Received --adversarial-eps-growth-epoch and --early-stopping together.')
        elif kwargs['adversarial_eps_growth_start'] is not None:
            logger.warning('Received --adversarial-eps-growth-start without --adversarial-eps-growth-epoch.')

        attack_config = utils.read_attack_config_file(
            kwargs['attack_config_file'])

        adversarial_attack = parsing.parse_attack_pool(
            kwargs['adversarial_training'], kwargs['domain'], kwargs['adversarial_p'], 'training', model, attack_config, kwargs['device'], seed=kwargs['seed'])

    # RS loss
    if kwargs['rs_regularization'] == 0:
        if kwargs['rs_eps'] is not None:
            logger.warning('Received --rs-eps without --rs-regularization.')
        if kwargs['rs_start_epoch'] != 1:
            logger.warning('Received --rs-start_epoch without --rs-regularization.')
    else:
        if kwargs['rs_eps'] is None:
            raise click.BadOptionUsage(
                '--rs-eps', 'Please specify the maximum perturbation for RS loss with --rs-eps.')

        if kwargs['rs_start_epoch'] > kwargs['epochs']:
            logger.warning('--rs-start-epoch is higher than the number of epochs. This means that RS loss will never be activated.')

        if kwargs['rs_start_epoch'] > 1 and kwargs['early_stopping'] > 0:
            logger.warning('Received --rs-start-epoch and --early-stopping together.')

    # Use Mean Cross Entropy, consistent with Xiao and Madry's ReLU training technique
    loss = torch.nn.CrossEntropyLoss(reduction='mean')
    optimiser = parsing.parse_optimiser(
        kwargs['optimiser'], model.parameters(), kwargs)

    if kwargs['checkpoint_every'] is None:
        checkpoint_path = None
    else:
        checkpoint_path = kwargs['save_to'] + '-checkpoint'

    if kwargs['load_checkpoint'] is None:
        loaded_checkpoint = None
    else:
        loaded_checkpoint = utils.torch_load(kwargs['load_checkpoint'])

    training.train(model, train_dataloader, optimiser, loss, kwargs['epochs'], kwargs['device'],
                      val_loader=val_dataloader, l1_regularization=kwargs['l1_regularization'],
                      rs_regularization=kwargs['rs_regularization'], rs_eps=kwargs['rs_eps'], rs_minibatch_size=kwargs['rs_minibatch'],
                      rs_start_epoch=kwargs['rs_start_epoch'], early_stopping=early_stopping, attack=adversarial_attack,
                      attack_ratio=kwargs['adversarial_ratio'], attack_p=kwargs['adversarial_p'],
                      attack_eps=kwargs['adversarial_eps'], attack_eps_growth_epoch=kwargs['adversarial_eps_growth_epoch'],
                      attack_eps_growth_start=kwargs['adversarial_eps_growth_start'],
                      checkpoint_every=kwargs['checkpoint_every'], checkpoint_path=checkpoint_path,
                      loaded_checkpoint=loaded_checkpoint, choose_best=kwargs['choose_best'])

    save_to = kwargs['save_to']
    pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_to)
