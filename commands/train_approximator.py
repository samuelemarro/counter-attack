import logging

import click
import pathlib
import torch

import adversarial_dataset as ad
import parsing
import training
import utils

logger = logging.getLogger(__name__)

# Nota: keep_misclassified viene ignorato per gli adversarial examples, dato che per definizione vengono misclassificati


@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('epochs', type=click.IntRange(1, None))
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
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
@click.option('--from-adversarial-dataset', is_flag=True, help='Use an adversarial dataset to compute the adversarial distance.')
@click.option('--val-from-adversarial-dataset', is_flag=True)
@click.option('--seed', type=int, default=None,
              help='The seed for random generation. If unspecified, the current time is used as seed.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
              help='The minimum logging level.')
@parsing.add_options(parsing.training_options)
def train_approximator(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])

    if kwargs['seed'] is not None:
        utils.set_seed(kwargs['seed'])

    model = parsing.parse_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'],
                              True, kwargs['masked_relu'], True, load_weights=False, as_detector=True)
    model.train()

    train_dataset = parsing.parse_dataset(
        kwargs['domain'], kwargs['dataset'], allow_standard=False)
    val_dataset = None

    if kwargs['from_adversarial_dataset']:
        # TODO: Controllare
        train_dataset = train_dataset.to_distance_dataset()
    elif isinstance(train_dataset, ad.AdversarialDataset):
        raise click.BadArgumentUsage('Expected a distance dataset as training dataset, got an adversarial dataset. '
                                     'If this is intentional, use --from-adversarial-dataset .')

    val_dataloader = None
    if kwargs['validation_split'] != 0:
        train_dataset, val_dataset = training.split_dataset(
            train_dataset, kwargs['validation_split'], shuffle=True)
    elif kwargs['validation_dataset'] is not None:
        val_dataset = parsing.parse_dataset(
            kwargs['domain'], kwargs['validation_dataset'], allow_standard=False)

        if kwargs['val_from_adversarial_dataset']:
            # TODO: controllare
            val_dataset = val_dataset.to_distance_dataset()
        elif isinstance(val_dataset, ad.AdversarialDataset):
            raise click.BadArgumentUsage('Expected a distance dataset as validation dataset, got an adversarial dataset. '
                                         'If this is intentional, use --val-from-adversarial-dataset .')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, kwargs['batch_size'], shuffle=kwargs['shuffle'])

    if val_dataset is None:
        val_dataloader = None
    else:
        # There is no point in shuffling the validation dataset
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, kwargs['batch_size'], shuffle=False)

    early_stopping = None
    if kwargs['early_stopping'] > 0:
        early_stopping = training.EarlyStopping(
            kwargs['early_stopping'], delta=kwargs['early_stopping_delta'])

    # TODO: Mean or Sum?
    loss = torch.nn.MSELoss()
    optimiser = parsing.parse_optimiser(
        kwargs['optimiser'], model.parameters(), kwargs)

    if kwargs['checkpoint_every'] is None:
        checkpoint_path = None
    else:
        checkpoint_path = kwargs['save_to'] + '-checkpoint'

    if kwargs['load_checkpoint'] is None:
        loaded_checkpoint = None
    else:
        loaded_checkpoint = torch.load(kwargs['load_checkpoint'])

    training.train(model, train_dataloader, optimiser, loss, kwargs['epochs'], kwargs['device'],
                      val_loader=val_dataloader, l1_regularization=kwargs['l1_regularization'], early_stopping=early_stopping,
                      checkpoint_every=kwargs['checkpoint_every'], checkpoint_path=checkpoint_path,
                      loaded_checkpoint=loaded_checkpoint)

    save_to = kwargs['save_to']
    pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_to)
