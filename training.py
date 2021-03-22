import logging
import os
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

import torch_utils
import utils

logger = logging.getLogger(__name__)

def conv_to_matrix(conv, image_shape, output_shape, device):
    identity = torch.eye(np.prod(image_shape).item()).reshape(
        [-1] + list(image_shape)).to(device)
    output = F.conv2d(identity, conv.weight, None, conv.stride, conv.padding)
    W = output.reshape(-1, np.prod(output_shape).item())
    # In theory W should be transposed, but the algorithm requires it to be left as it is
    b = torch.stack(
        [torch.ones(output_shape[1:], device=device) * bi for bi in conv.bias])
    #b = b.reshape(-1, np.prod(output_shape[1:]).item())
    b = b.reshape(-1)
    # print(b.shape)
    # print(conv.bias.shape)
    # print('===')

    return W, b


def _interval_arithmetic(lb, ub, W, b):
    W_max = torch.maximum(W, torch.tensor(0.0).to(W))
    W_min = torch.minimum(W, torch.tensor(0.0).to(W))
    new_lb = torch.matmul(lb, W_max) + torch.matmul(ub, W_min) + b
    new_ub = torch.matmul(ub, W_max) + torch.matmul(lb, W_min) + b
    return new_lb, new_ub

# Assumes shapes of m, m, Bxmxn, n


def _interval_arithmetic_batch(lb, ub, W, b):
    W_max = torch.maximum(W, torch.tensor(0.0).to(W))
    W_min = torch.minimum(W, torch.tensor(0.0).to(W))
    new_lb = torch.einsum("m,bmn->bn", lb, W_max) + \
        torch.einsum("m,bmn->bn", ub, W_min) + b
    new_ub = torch.einsum("m,bmn->bn", ub, W_max) + \
        torch.einsum("m,bmn->bn", lb, W_min) + b
    return new_lb, new_ub

# Assumes shapes of Bxm, Bxm, Bxmxn, Bxn


def _interval_arithmetic_all_batch(lb, ub, W, b):
    W_max = torch.maximum(W, torch.tensor(0.0).to(W))
    W_min = torch.minimum(W, torch.tensor(0.0).to(W))
    new_lb = torch.einsum("bm,bmn->bn", lb, W_max) + \
        torch.einsum("bm,bmn->bn", ub, W_min) + b
    new_ub = torch.einsum("bm,bmn->bn", ub, W_max) + \
        torch.einsum("bm,bmn->bn", lb, W_min) + b
    return new_lb, new_ub


def _compute_bounds_n_layers(n, lbs, ubs, Ws, biases):
    # print(n, len(lbs))
    assert n == len(lbs)
    assert n == len(ubs)
    assert n == len(Ws)
    assert n == len(biases)

    # Current layer
    lb = lbs[0]
    ub = ubs[0]
    W = Ws[0]
    b = biases[0]

    #print('n: ', n)
    # Base case
    if n == 1:
        if len(W.shape) == 2:
            naive_ia_bounds = _interval_arithmetic(lb, ub, W, b)
        else:
            naive_ia_bounds = _interval_arithmetic_all_batch(lb, ub, W, b)
        return naive_ia_bounds

    # Recursive case
    W_prev = Ws[1]
    b_prev = biases[1]

    # Compute W_A and W_NA
    out_dim = W.shape[-1]
    active_mask_unexpanded = (lb > 0).float()

    # active_mask = torch.tile(torch.unsqueeze(active_mask_unexpanded, 2), [1, 1, out_dim]) # This should be B x y x p
    active_mask = torch.unsqueeze(active_mask_unexpanded, 2).expand(
        [-1, -1, out_dim])  # This should be B x y x p

    nonactive_mask = 1.0 - active_mask
    #print('W: ', W.shape)
    #print('active_mask: ', active_mask.shape)
    W_A = torch.mul(W, active_mask)  # B x y x p
    W_NA = torch.mul(W, nonactive_mask)  # B x y x p

    # Compute bounds from previous layer
    if len(lb.shape) == 2:
        prev_layer_bounds = _interval_arithmetic_all_batch(lb, ub, W_NA, b)
    else:
        prev_layer_bounds = _interval_arithmetic_batch(
            lb, ub, W_NA, b)  # TODO: Quando avviene?

    # Compute new products
    W_prod = torch.einsum('my,byp->bmp', W_prev, W_A)  # b x m x p
    b_prod = torch.einsum('y,byp->bp', b_prev, W_A)  # b x p

    #print('W_prod: ', W_prod.shape)
    #print('b_prod: ', b_prod.shape)

    lbs_new = lbs[1:]
    ubs_new = ubs[1:]
    Ws_new = [W_prod] + Ws[2:]
    biases_new = [b_prod] + biases[2:]

    deeper_bounds = _compute_bounds_n_layers(
        n-1, lbs_new, ubs_new, Ws_new, biases_new)
    return (prev_layer_bounds[0] + deeper_bounds[0], prev_layer_bounds[1] + deeper_bounds[1])


def model_to_rs_sequence(model, input_shape, device):
    layers = torch_utils.unpack_sequential(model)
    new_layers = []

    placeholder = torch.zeros([1] + list(input_shape), device=device)

    logger.debug('[RS Loss] Parsed layers:')
    for layer in layers:
        if isinstance(layer, torch_utils.Normalisation) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.Flatten):
            logger.debug('[RS Loss] Non-reshaping layer of type %s, inserting as-is.', type(layer).__name__)
            placeholder = layer(placeholder)
            new_layers.append(layer)
        elif isinstance(layer, nn.Conv2d):
            logger.debug('[RS Loss] 2D convolution, replacing with linear layer.')
            before_conv_shape = placeholder.shape[1:]
            placeholder = layer(placeholder)
            after_conv_shape = placeholder.shape[1:]
            W, b = conv_to_matrix(layer, before_conv_shape,
                                  after_conv_shape, device)
            new_layers.append((W, b))
        elif isinstance(layer, nn.Linear):
            logger.debug('[RS Loss] Linear layer, inserting as-is.')
            placeholder = layer(placeholder)
            new_layers.append((layer.weight.T, layer.bias))
        else:
            raise NotImplementedError(
                f'Unsupported layer {type(layer).__name__}.')

    return new_layers


def rs_loss(model, x, epsilon, input_min=0, input_max=1):
    layers = model_to_rs_sequence(model, x.shape[1:], x.device)
    batch_size = x.shape[0]
    total_loss = 0

    input_lower = torch.clamp(x - epsilon, min=input_min, max=input_max)
    input_upper = torch.clamp(x + epsilon, min=input_min, max=input_max)

    if isinstance(layers[0], torch_utils.Normalisation):
        logger.debug('[RS Loss] Applying normalisation')
        input_lower = layers[0].forward(input_lower)
        input_upper = layers[0].forward(input_upper)
        layers = layers[1:]

    input_lower = input_lower.reshape(batch_size, -1)
    input_upper = input_upper.reshape(batch_size, -1)

    post_relu_lowers = []
    post_relu_uppers = []
    Ws = []
    bs = []

    post_relu_lowers.append(input_lower)
    post_relu_uppers.append(input_upper)

    # RS Loss is designed for networks that are sequences of conv/linear and ReLUs
    layer_index = 0

    for layer in layers:
        if isinstance(layer, torch_utils.Normalisation):
            raise RuntimeError(
                'More than one normalisation in the Sequential.')
        elif isinstance(layer, nn.ReLU):
            lower = F.relu(lower)
            upper = F.relu(upper)
            post_relu_lowers.insert(0, lower)
            post_relu_uppers.insert(0, upper)
        elif isinstance(layer, tuple):
            layer_index += 1
            W, b = layer

            Ws.insert(0, W)
            bs.insert(0, b)
            #print(f'layer {layer_index}: {W.shape}, {b.shape}')

            if len(post_relu_lowers) != layer_index:
                raise RuntimeError('There aren\'t as many Linear/Conv2D layers as ReLU layers. '
                                   'Check the architecture of the model.')

            lower, upper = _compute_bounds_n_layers(
                layer_index, post_relu_lowers, post_relu_uppers, Ws, bs)

            # Il segno è corretto?
            total_loss -= torch.mean(torch.sum(torch.tanh(1 +
                                                          lower * upper), -1))
        elif not isinstance(layer, nn.Flatten):  # Flatten is ignored
            raise NotImplementedError('Unsupported layer')

    #print('Total loss:', total_loss)
    return total_loss

def train(model, train_loader, optimiser, loss_function, max_epochs, device, val_loader=None,
          l1_regularization=0, rs_regularization=0, rs_eps=0, rs_minibatch=None, rs_start_epoch=0,
          early_stopping=None, attack=None, attack_ratio=0.5, attack_p=None, attack_eps=None,
          attack_eps_growth_epoch=0, attack_eps_growth_start=None, checkpoint_every=None, checkpoint_path=None,
          loaded_checkpoint=None):
    if early_stopping is not None:
        if val_loader is None:
            raise ValueError('Early stopping requires a validation loader.')
    if attack_eps is not None and attack_eps_growth_start is not None and attack_eps_growth_start > attack_eps:
        raise ValueError('attack_eps_growth_start should be smaller than or equal to rs_eps.')
    if (checkpoint_every is None) ^ (checkpoint_path is None):
        raise ValueError('checkpoint_every and checkpoint_path should be either both None or both not None.')

    # Prepare the epsilon values
    if attack_eps_growth_epoch == 0:
        epsilons = [attack_eps] * max_epochs
    else:
        epsilons = list(np.linspace(attack_eps_growth_start, attack_eps, num=attack_eps_growth_epoch))
        epsilons += list([attack_eps] * (max_epochs - attack_eps_growth_epoch))

    model.train()
    model.to(device)
    iterator = tqdm(range(max_epochs), desc='Training')

    if loaded_checkpoint is None:
        start_epoch = 0
    else:
        start_epoch = loaded_checkpoint['epoch'] + 1
        model.load_state_dict(loaded_checkpoint['model'])
        optimiser.load_state_dict(loaded_checkpoint['optimiser'])

        if (early_stopping is None) ^ (loaded_checkpoint['early_stopping'] is None):
            logger.warning('There is a mismatch between the current early_stopping and '
                           'the saved one.')

        if early_stopping is not None and loaded_checkpoint['early_stopping'] is not None:
            early_stopping.load_state_dict(loaded_checkpoint['early_stopping'])

    for epoch in iterator:
        if epoch < start_epoch:
            continue

        epsilon = epsilons[epoch]

        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)

            if attack is not None:
                indices = np.random.choice(
                    list(range(len(x))), int(len(x) * attack_ratio))
                adversarial_x = x[indices]
                adversarial_targets = target[indices]

                logger.debug('Disabling model parameter gradients.')
                restore_list = torch_utils.disable_model_gradients(model)

                logger.debug('Running adversarial attack with epsilon %s.', epsilon)
                adversarials = attack.perturb(
                    adversarial_x, y=adversarial_targets, eps=epsilon).detach()
                
                logger.debug('Restoring model parameter gradients.')
                torch_utils.restore_model_gradients(model, restore_list)
                del restore_list

                # In Madry's original paper on adversarial training, the authors do not check
                # the success of the attack: they just clip the resulting adversarial to the allowed
                # input range
                adversarials = utils.clip_adversarial(adversarials, adversarial_x, epsilon, input_min=0, input_max=1)

                for j, index in enumerate(indices):
                    if adversarials[j] is not None:
                        x[index] = adversarials[j]
            y_pred = model(x)

            # Adversarial loss isn't divided by the batch size
            loss = loss_function(y_pred, target)

            if l1_regularization != 0:
                for group in optimiser.param_groups:
                    for p in group['params']:
                        # L1 loss isn't divided by the batch size
                        loss += torch.sum(torch.abs(p)) * l1_regularization

            optimiser.zero_grad()
            loss.backward()

            # RS Regularization uses a high amount of GPU memory, so we use .backward()
            # for each minibatch
            if rs_regularization != 0 and (epoch + 1) >= rs_start_epoch:
                if rs_minibatch is None:
                    rs = rs_loss(model, x, epsilon=rs_eps) * rs_regularization
                    rs.backward()
                else:
                    for minibatch in torch_utils.split_batch(x, rs_minibatch):
                        # RS loss isn't divided by the batch size
                        rs = rs_loss(model, minibatch,
                                     epsilon=rs_eps) * rs_regularization
                        rs.backward()

            optimiser.step()

        if val_loader is not None:
            logger.debug('Computing validation loss.')
            val_loss = 0
            with torch.no_grad():
                #rs_sum = 0
                for x_val, target_val in val_loader:
                    x_val = x_val.to(device)
                    target_val = target_val.to(device)

                    y_pred_val = model(x_val)
                    val_loss += loss_function(y_pred_val, target_val)

                    # TODO: In realtà può essere calcolato una sola volta, anche se così facendo scombina il suo peso relativo
                    if l1_regularization != 0:
                        for group in optimiser.param_groups:
                            for p in group['params']:
                                val_loss += torch.sum(torch.abs(p)) * \
                                    l1_regularization

                    if rs_regularization != 0:
                        if rs_minibatch is None:
                            rs = rs_loss(model, x, epsilon=rs_eps) * \
                                rs_regularization
                            val_loss += rs
                        else:
                            for minibatch in torch_utils.split_batch(x, rs_minibatch):
                                rs = rs_loss(model, minibatch,
                                             epsilon=rs_eps) * rs_regularization
                                val_loss += rs

            iterator.set_description('Training | Validation Loss: {:.3e}'.format(
                val_loss.cpu().detach().item()))

            if early_stopping is not None:
                early_stopping(val_loss, model)

                if early_stopping.stop:
                    logger.debug('Early stop triggered.')
                    break

        if (checkpoint_path is not None) and (epoch + 1) % checkpoint_every == 0:
            if not pathlib.Path(checkpoint_path).exists():
                os.mkdir(checkpoint_path)
            
            # Note: We use 1-indexing for epochs
            current_epoch_path = pathlib.Path(checkpoint_path) / f'{epoch + 1}.check'
            torch.save({
                'optimiser' : optimiser.state_dict(),
                'epoch' : epoch,
                'model' : model.state_dict(),
                'early_stopping' : None if early_stopping is None else early_stopping.state_dict()
            }, current_epoch_path)

        # If you are using early_stopping, load the best model
        if early_stopping is not None:
            logger.debug('Early stopping: Loading best state dict.')
            model.load_state_dict(early_stopping.best_state_dict)

class StartStopDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = len(dataset)

        if start < 0:
            raise ValueError('start must be at least 0.')
        if stop > len(dataset):
            raise ValueError(
                'stop must be smaller than or equal to the dataset size.')
        if stop <= start:
            raise ValueError('stop must be strictly larger than start.')

        self.dataset = dataset
        self.start = start
        self.stop = stop

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if self.start + idx.stop > self.stop:
                raise ValueError('Slice stop is bigger than dataset stop.')
            slice_ = slice(self.start + idx.start,
                           self.start + idx.stop, idx.step)
            return self.dataset[slice_]
        else:
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

    def state_dict(self):
        return {
            'patience' : self.patience,
            'counter' : self.counter,
            'best_score' : self.best_score,
            'stop' : self.stop,
            'val_loss_min' : self.val_loss_min,
            'delta' : self.delta,
            'best_state_dict' : self.best_state_dict
        }
    
    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.stop = state_dict['stop']
        self.val_loss_min = state_dict['val_loss_min']
        self.delta = state_dict['delta']
        self.best_state_dict = state_dict['best_state_dict']
