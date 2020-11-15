import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm

import utils

import logging
logger = logging.getLogger(__name__)

class BatchLimitedModel(nn.Module):
    def __init__(self, wrapped_model, batch_size):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.batch_size = batch_size

    def forward(self, x):
        num_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))

        outputs = []

        for batch_id in range(num_batches):
            batch_begin = batch_id * self.batch_size
            batch_end = (batch_id + 1) * self.batch_size

            batch = x[batch_begin:batch_end]
            
            outputs.append(self.wrapped_model(batch))

        outputs = torch.cat(outputs)

        assert len(outputs) == len(x)

        return outputs

class Normalisation(nn.Module):
    def __init__(self, mean, std, num_channels=3):
        super().__init__()
        self.mean = torch.from_numpy(np.array(mean).reshape((num_channels, 1, 1)))
        self.std = torch.from_numpy(np.array(std).reshape((num_channels, 1, 1)))

    def forward(self, input):
        mean = self.mean.to(input)
        std = self.std.to(input)
        return (input - mean) / std

# Modular version of torch.squeeze()
class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, self.dim)

# TODO: Controlla

# A ReLU module where some ReLU calls are replaced with fixed behaviour
# (zero or linear)
class MaskedReLU(nn.Module):
    def __init__(self, mask_shape):
        super().__init__()
        self.always_linear = nn.Parameter(torch.zeros(mask_shape, dtype=torch.bool), requires_grad=False)
        self.always_zero = nn.Parameter(torch.zeros(mask_shape, dtype=torch.bool), requires_grad=False)

    def forward(self, x):
        # Note: We do not use actual masking because
        # that would require using boolean indexing, which
        # causes a CUDA synchronization (causing major slowdowns)
        
        output = torch.relu(x)

        # always_zero masking
        output = utils.fast_boolean_choice(output, 0, self.always_zero)

        # always_linear masking
        output = utils.fast_boolean_choice(output, x, self.always_linear)

        return output

class ReLUCounter(nn.ReLU):
    def __init__(self):
        super().__init__()
        self.positive_counter = None
        self.negative_counter = None

    def forward(self, x):
        if self.positive_counter is None:
            self.positive_counter = torch.zeros(x.shape[1:], dtype=torch.long, device=x.device)
            self.negative_counter = torch.zeros(x.shape[1:], dtype=torch.long, device=x.device)

        positive = (x > 0).long().sum(dim=0)
        negative = (x < 0).long().sum(dim=0)

        self.positive_counter += positive
        self.negative_counter += negative

        return torch.relu(x)
        

# For adversarial training, we don't replace genuines with failed adversarial samples

def train(model, train_loader, optimiser, loss_function, max_epochs, device, val_loader=None, l1_regularization=0, early_stopping=None, attack=None, attack_ratio=0.5, attack_p=None, attack_eps=None):
    if early_stopping is not None:
        if val_loader is None:
            raise ValueError('Early stopping requires a validation loader.')

    model.train()
    model.to(device)

    iterator = tqdm(range(max_epochs), desc='Training')

    for i in iterator:
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)

            if attack is not None:
                indices = np.random.choice(list(range(len(x))), int(len(x) * attack_ratio))
                adversarial_x = x[indices]
                adversarial_targets = target[indices]

                adversarials = attack.perturb(adversarial_x, y=adversarial_targets).detach()

                adversarials = utils.remove_failed(model, adversarial_x, adversarial_targets, adversarials, False, p=attack_p, eps=attack_eps)

                for i, index in enumerate(indices):
                    if adversarials[i] is not None:
                        x[index] = adversarials[i]

            y_pred = model(x)

            loss = loss_function(y_pred, target)

            if l1_regularization != 0:
                for group in optimiser.param_groups:
                    for p in group['params']:
                        loss += torch.sum(torch.abs(p)) * l1_regularization

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if val_loader is not None:
            val_loss = 0
            with torch.no_grad():
                for x_val, target_val in val_loader:
                    x_val = x_val.to(device)
                    target_val = target_val.to(device)

                    y_pred_val = model(x_val)
                    val_loss += loss_function(y_pred_val, target_val)

            iterator.set_description('Training | Validation Loss: {:.3e}'.format(val_loss.cpu().detach().item()))

            if early_stopping is not None:
                early_stopping(val_loss, model)

                if early_stopping.stop:
                    model.load_state_dict(early_stopping.best_state_dict)
                    break

        # In case the validation loss did not improve but the training
        # reached the max number of epochs, load the best model
        if early_stopping is not None:
            model.load_state_dict(early_stopping.best_state_dict)


def unpack_sequential(module):
    layers = []
    for layer in module:
        if isinstance(layer, nn.Sequential):
            layers += unpack_sequential(layer)
        else:
            layers.append(layer)

    return layers

class StartStopDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = len(dataset)

        if start < 0:
            raise ValueError('start must be at least 0.')
        if stop > len(dataset):
            raise ValueError('stop must be smaller than or equal to the dataset size.')
        if stop <= start:
            raise ValueError('stop must be strictly larger than start.')

        self.dataset = dataset
        self.start = start
        self.stop = stop

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]

    def __len__(self):
        return self.stop - self.start

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        assert len(indices) <= len(dataset)
        assert max(indices) < len(dataset)
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0     
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_state_dict = None
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        self.best_state_dict = model.state_dict()
        self.val_loss_min = val_loss

def split_dataset(original_dataset, val_split, shuffle=True):
    dataset_size = len(original_dataset)
    indices = list(range(dataset_size))
    split_index = int(np.floor(val_split * dataset_size))

    if shuffle:
        np.random.shuffle(indices)

    val_indices, train_indices = indices[:split_index], indices[split_index:]

    train_dataset = IndexedDataset(original_dataset, train_indices)
    val_dataset = IndexedDataset(original_dataset, val_indices)

    return train_dataset, val_dataset