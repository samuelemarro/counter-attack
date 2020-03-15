import logging

import click
import torch

import parsing
import tests

logger = logging.getLogger(__name__)

@click.command()
@click.argument('domain', type=click.Choice(parsing.domains))
@click.argument('architecture', type=click.Choice(parsing.architectures))
@click.argument('dataset')
@click.option('--state-dict-path', type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True)
@click.option('--device', default='cuda', show_default=True)
@click.option('--from-adversarial-dataset', is_flag=True)
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True)
def accuracy(**kwargs):
    parsing.set_log_level(kwargs['log_level'])
    
    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'])

    if kwargs['from_adversarial_dataset']:
        dataset = dataset.to_adversarial_training_dataset()
        logger.warning('The accuracy will be computed only on the successful adversarial examples.')

    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)
    
    accuracy = tests.accuracy(model, dataloader, kwargs['device'])

    print('Accuracy: {:.2f}%'.format(accuracy * 100.0))