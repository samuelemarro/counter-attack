import logging

import click
import ignite
import pathlib
import torch

import parsing
import tests
import torch_utils

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.argument('epochs', type=click.IntRange(1, None))
@click.argument('save_to', type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None,
    help='The path to the state-dict file of the model. If None, a pretrained model will be used (if available).')
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
    help='The batch size of the dataset.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@parsing.add_options(parsing.training_options)
@click.option('--seed', type=int, default=None,
    help='The seed for random generation. If unspecified, the current time is used as seed.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
def train_classifier(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['seed'] is not None:
        torch.manual_seed(kwargs['seed'])
    
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