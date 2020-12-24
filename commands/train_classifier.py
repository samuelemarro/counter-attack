import logging

import click
import ignite
import pathlib
import torch
import torchvision

import utils
import parsing
import tests
import torch_utils

logger = logging.getLogger(__name__)

# TODO: Non ha senso che di default carichi un modello pretrained
@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('epochs', type=click.IntRange(1, None))
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
# TODO: Rimuovere --state-dict-path (tanto è inutile)
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the state-dict file of the model. If None, a pretrained model will be used (if available).')
@click.option('--masked-relu', is_flag=True,
    help='If passed, all ReLU layers will be converted to MaskedReLU layers.')
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
    help='The batch size of the dataset.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--cpu-threads', type=click.IntRange(1, None, False), default=None,
    help='The number of PyTorch CPU threads. If unspecified, the default '
          'number is used (usually the number of cores).')
@parsing.add_options(parsing.training_options)
@click.option('--rs-regularization', type=float, default=0, show_default=True,
        help='The weight of RS regularization. 0 disables RS regularization. If enabled, requires specifying --rs-eps.')
@click.option('--rs-eps', type=float, default=None,
        help='The epsilon of RS regularization. Ignored if RS regularization is enabled. '
             'Required if RS regularization is enabled.')
@click.option('--rs-minibatch', type=click.IntRange(1, None), default=None,
        help='If enabled, the batch will be split into minibatches of size rs_minibatch before being used to '
             'compute the RS loss. This helps to reduce GPU memory usage. Ignored if RS regularization is disabled.')
@click.option('--flip', is_flag=True,
    help='Enables random horizontal flipping.')
@click.option('--rotation', type=float, default=0, show_default=True,
    help='Random rotation (in degrees) in range (-value, +value).')
@click.option('--translation', type=float, default=0, show_default=True,
    help='Random horizontal and vertical translation in range (-value * image_size, +value * image_size).')
@click.option('--adversarial-training', type=click.Choice(parsing.supported_attacks), default=None,
    help='The adversarial attack that will be used to compute the adversarials. '
         'If unspecified, disables adversarial training. Requires specifying --adversarial-ratio, --adversarial-p '
         'and --adversarial-eps.')
@click.option('--adversarial-p', type=click.Choice(parsing.distances), callback=parsing.validate_lp_distance, default=None,
    help='The Lp norm used for adversarial training. '
         'Ignored if adversarial training is disabled. '
         'Required if adversarial training is enabled.')
@click.option('--adversarial-ratio', type=float, default=None,
    help='The ratio of samples that are replaced with adversarials. '
         'Ignored if adversarial training is disabled. '
         'Requird if adversarial training is enabled.')
@click.option('--adversarial-eps', type=float, default=None,
    help='The maximum perturbation of an adversarial. '
         'Ignored if adversarial training is disabled. '
         'Required if adversarial training is enabled.')
@click.option('--adversarial-cfg-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='default_attack_configuration.cfg', show_default=True,
    help='The path to the attack configuration file for adversarial training. Ignored if adversarial training is disabled.')
@click.option('--seed', type=int, default=None,
    help='The seed for random generation. If unspecified, the current time is used as seed.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
def train_classifier(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])

    if kwargs['seed'] is not None:
        torch.manual_seed(kwargs['seed'])
    
    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, kwargs['masked_relu'], load_weights=False)
    model.train()

    extra_transforms = []

    if kwargs['flip']:
        extra_transforms.append(torchvision.transforms.RandomHorizontalFlip())
    
    if kwargs['rotation'] != 0 or kwargs['translation'] != 0:
        translation = (kwargs['translation'], kwargs['translation']) if kwargs['translation'] != 0 else None
        extra_transforms.append(torchvision.transforms.RandomAffine(kwargs['rotation'], translation))

    train_dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], extra_transforms=extra_transforms)
    val_dataset = None

    if kwargs['validation_dataset'] is not None and kwargs['validation_split'] != 0:
        raise click.BadOptionUsage('--validation_split', '--validation_dataset and validation_split are mutually exclusive.')

    if kwargs['validation_split'] != 0:
        train_dataset, val_dataset = torch_utils.split_dataset(train_dataset, kwargs['validation_split'], shuffle=True)
    elif kwargs['validation_dataset'] is not None:
        val_dataset = parsing.get_dataset(kwargs['domain'], kwargs['validation_dataset'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=kwargs['shuffle'])
    if val_dataset is None:
        val_dataloader = None
    else:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, kwargs['batch_size'], shuffle=False)

    early_stopping = None
    if kwargs['early_stopping'] > 0:
        early_stopping = torch_utils.EarlyStopping(kwargs['early_stopping'], delta=kwargs['early_stopping_delta'])

    # TODO: Prendere adversarial_training come 0 = None?
    if kwargs['adversarial_training'] is None:
        adversarial_attack = None
    else:
        if kwargs['adversarial_training'] == 'mip':
            raise click.BadOptionUsage('--adversarial-training', 'adversarial training currently does not support MIP.')

        if kwargs['adversarial_ratio'] is None:
            raise click.BadOptionUsage('--adversarial-ratio', 'Please specify the ratio for adversarial training with --adversarial-ratio .')

        if kwargs['adversarial_p'] is None:
            raise click.BadOptionUsage('--adversarial-p', 'Please specify the Lp norm for adversarial training with --adversarial-p .')

        if kwargs['adversarial_eps'] is None:
            raise click.BadOptionUsage('--adversarial-eps', 'Please specify the maximum perturbarion for adversarial training with --adversarial-eps (inf is also allowed).')

        attack_config = utils.read_attack_config_file(kwargs['adversarial_cfg_file'])

        adversarial_attack = parsing.get_attack(kwargs['adversarial_training'], kwargs['domain'], kwargs['adversarial_p'], 'training', model, attack_config)

    if kwargs['rs_regularization'] != 0 and kwargs['rs_eps'] is None:
        raise click.BadOptionUsage('--rs-eps', 'Please specify the maximum perturbation for RS loss with --rs-eps.')

    loss = torch.nn.CrossEntropyLoss()
    optimiser = parsing.get_optimiser(kwargs['optimiser'], model.parameters(), kwargs)

    torch_utils.train(model, train_dataloader, optimiser, loss, kwargs['epochs'], kwargs['device'],
    val_loader=val_dataloader, l1_regularization=kwargs['l1_regularization'],
    rs_regularization=kwargs['rs_regularization'], rs_eps=kwargs['rs_eps'], rs_minibatch=kwargs['rs_minibatch'],
    early_stopping=early_stopping, attack=adversarial_attack,
    attack_ratio=kwargs['adversarial_ratio'], attack_eps=kwargs['adversarial_eps'],
    attack_p=kwargs['adversarial_p'])

    save_to = kwargs['save_to']
    pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_to)