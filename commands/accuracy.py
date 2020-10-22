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
@click.option('--from-adversarial-dataset', is_flag=True,
    help='If passed, the accuracy is computed on an adversarial dataset.')
@click.option('--max-samples', type=click.IntRange(1, None), default=None,
    help='The maximum number of images that are loaded from the dataset. '
         'If unspecified, all images are loaded.')
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
def accuracy(**kwargs):
    parsing.set_log_level(kwargs['log_level'])

    if kwargs['cpu_threads'] is not None:
        torch.set_num_threads(kwargs['cpu_threads'])

    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, kwargs['masked_relu'], load_weights=True)
    model.eval()

    dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], max_samples=kwargs['max_samples'])

    if kwargs['from_adversarial_dataset']:
        dataset = dataset.to_adversarial_training_dataset()
        logger.warning('The accuracy will be computed only on the successful adversarial examples.')

    dataloader = torch.utils.data.DataLoader(dataset, kwargs['batch_size'], shuffle=False)
    
    accuracy = tests.accuracy(model, dataloader, kwargs['device'])

    print('Accuracy: {:.2f}%'.format(accuracy * 100.0))