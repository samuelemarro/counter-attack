import logging

import click
import pathlib
import torch

import adversarial_dataset as ad
import parsing
import tests
import torch_utils

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
@click.option('--batch-size', type=click.IntRange(1), default=50, show_default=True,
    help='The batch size of the dataset.')
@click.option('--device', default='cuda', show_default=True, help='The device where the model will be executed.')
@click.option('--from-adversarial-dataset', is_flag=True, help='Use an adversarial dataset to compute the adversarial distance.')
@click.option('--val-from-adversarial-dataset', is_flag=True)
@click.option('--log-level', type=click.Choice(parsing.log_levels), default='info', show_default=True,
    help='The minimum logging level.')
@parsing.add_options(parsing.training_options)
def train_approximator(**kwargs):
    parsing.set_log_level(kwargs['log_level'])
    
    model = parsing.get_model(kwargs['domain'], kwargs['architecture'], kwargs['state_dict_path'], True, load_weights=False, as_detector=True)
    model.train()

    train_dataset = parsing.get_dataset(kwargs['domain'], kwargs['dataset'], allow_standard=False)

    if kwargs['from_adversarial_dataset']:
        train_dataset = train_dataset.to_distance_dataset()
    elif isinstance(train_dataset, ad.AdversarialDataset):
        raise click.BadArgumentUsage('Expected a distance dataset as training dataset, got an adversarial dataset. '
                                    'If this is intentional, use --from-adversarial-dataset .')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=kwargs['shuffle'])

    val_dataloader = None
    if kwargs['validation_dataset'] is not None:
        val_dataset = parsing.get_dataset(kwargs['domain'], kwargs['validation_dataset'], allow_standard=False)

        if kwargs['val_from_adversarial_dataset']:
            val_dataset = val_dataset.to_distance_dataset()
        elif isinstance(val_dataset, ad.AdversarialDataset):
            raise click.BadArgumentUsage('Expected a distance dataset as validation dataset, got an adversarial dataset. '
                                        'If this is intentional, use --val-from-adversarial-dataset .')

        # There's no point in shuffling the validation dataset
        val_dataloader = torch.utils.data.DataLoader(val_dataset, kwargs['batch_size'], shuffle=False)

    loss = torch.nn.MSELoss()
    optimiser = parsing.get_optimiser(kwargs['optimiser'], model.parameters(), kwargs)

    torch_utils.train(model, train_dataloader, optimiser, loss, kwargs['epochs'], kwargs['device'], val_loader=val_dataloader)

    save_to = kwargs['save_to']
    pathlib.Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_to)