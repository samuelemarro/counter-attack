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

def conv_to_matrix(conv, image_shape, output_shape, device, tf_weights):
    # Since a convolution is, at its essence, a matrix multiplication,
    # we can use conv(I) to compute W.T
    # Specifically, PyTorch computes matmul(I.T, W.T).T, which is equivalent
    # to matmul(W, I).T = W.T
    identity = torch.eye(np.prod(image_shape), device=device).reshape(
        [-1] + list(image_shape))

    # No bias (since it is computed separately)
    output = F.conv2d(identity, conv.weight, None, conv.stride, conv.padding)
    WT = output.reshape(-1, np.prod(output_shape))
    # In theory W should be transposed, but the algorithm requires it to be left as it is

    # b is the bias tensor repeated for every pixel of the output
    b = torch.stack(
        [torch.ones(output_shape[1:], device=device) * bi for bi in conv.bias])

    b = b.reshape(-1)

    # TODO: Capire come si relaziona la trasposizione con RS. E perché quelli dei layer vengono trasposti,
    # mentre questi no? Dopotutto anche qui i W non sono utilizzabili direttamente
    # In generale tutta questa questione della trasposizione non è ancora ben chiara. Per il momento quello che so è:
    # - Il peso di linear è direttamente utilizzabile, ma noi lo forniamo trasposto
    # - conv_to_matrix dà una W che funziona solo se trasposta
    # Quindi noi stiamo fornendo in entrambi i casi un peso trasposto!
    # Ma se gli forniamo quelli "giusti", _interval_arithmetic fallisce
    # Si può considerare di passare transpose=True

    # Salta fuori che Tensorflow calcola nei linear matmul(x, W)
    # dove W è [n_in, n_out], mentre PyTorch calcola
    # matmul(x, W^T) dove W è [n_out, n_in]
    # PyTorch è quindi equivalente a fare matmul(W, x), spiegando perché
    # posso usare direttamente il peso dei layer lineari
    # TODO: Rimuovere il testo in italiano

    # Wt has shape (in_channels, out_channels)

    if tf_weights:
        # Tensorflow accepts transposed weights (in_channels, out_channels)
        return WT, b
    else:
        # PyTorch accepts traditional weights (out_channels, in_channels)
        return WT.T, b


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


def model_to_linear_sequence(model, input_shape, device, tf_weights):
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
            logger.debug('[RS Loss] Before conv shape: %s, After conv shape: %s', before_conv_shape, after_conv_shape)
            W, b = conv_to_matrix(layer, before_conv_shape,
                                  after_conv_shape, device, tf_weights)
            new_layers.append((W, b))
        elif isinstance(layer, nn.Linear):
            logger.debug('[RS Loss] Linear layer, inserting as-is.')
            placeholder = layer(placeholder)

            if tf_weights:
                # Tensorflow accepts transposed weights (in_channels, out_channels)
                weight = layer.weight.T
            else:
                # PyTorch accepts traditional weights (out_channels, in_channels)
                weight = layer.weight

            new_layers.append((weight, layer.bias))
        else:
            raise NotImplementedError(
                f'Unsupported layer {type(layer).__name__}.')

    return new_layers


def rs_loss(model, x, epsilon, input_min=0, input_max=1):
    # TODO: TENERE A MENTE LAYER-FIRST VS LAYER-LAST PER PYTORCH VS TENSORFLOW
    # Use tf-like weights
    layers = model_to_linear_sequence(model, x.shape[1:], x.device, True)
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

def adversarial_training(x, target, model, attack, attack_ratio, epsilon):
    x = x.clone()
    target = target.clone()
    # Pick a portion of the samples (how many depends on attack_ratio)
    indices = np.random.choice(
        list(range(len(x))), int(len(x) * attack_ratio), replace=False)
    selected_x = x[indices]
    selected_targets = target[indices]

    logger.debug('Disabling model parameter gradients.')
    restore_list = torch_utils.disable_model_gradients(model)

    logger.debug('Running adversarial attack with epsilon %s.', epsilon)
    adversarials = attack.perturb(
        selected_x, y=selected_targets, eps=epsilon).detach()

    logger.debug('Restoring model parameter gradients.')
    torch_utils.restore_model_gradients(model, restore_list)

    # In Madry's original paper on adversarial training, the authors do not check
    # the success of the attack: they just clip the resulting adversarial to the allowed
    # input range
    adversarials = utils.clip_adversarial(adversarials, selected_x, epsilon, input_min=0, input_max=1)

    # Match adversarials with their original genuine
    for j, index in enumerate(indices):
        if adversarials[j] is not None:
            x[index] = adversarials[j]

    return x

# Following Xiao and Madry's implementation, l1 loss is computed by considering the
# convolutions as if they were their corresponding fully-connected matrices
def l1_loss(model, input_shape, device, l1_regularization):
    # Use standard weights
    layers = model_to_linear_sequence(model, input_shape, device, False)
    loss = 0
    for layer in layers:
        if isinstance(layer, tuple):
            # Only compute l1 loss on the weights
            W, _ = layer
            loss += torch.sum(torch.abs(W)) * l1_regularization

    return loss

# Note: Xiao and Madry's ReLU training technique also supports sparse weight initialization,
# which is however disabled by default

def train(model, train_loader, optimiser, loss_function, max_epochs, device, val_loader=None,
          l1_regularization=0, rs_regularization=0, rs_eps=0, rs_minibatch_size=None, rs_start_epoch=0,
          early_stopping=None, attack=None, attack_ratio=0.5, attack_p=None, attack_eps=None,
          attack_eps_growth_epoch=0, attack_eps_growth_start=None, checkpoint_every=None, checkpoint_path=None,
          loaded_checkpoint=None, choose_best=False):
    # Perform basic checks
    if early_stopping is not None and val_loader is None:
        raise ValueError('Early stopping requires a validation loader.')
    if attack_eps is not None and attack_eps_growth_start is not None and attack_eps_growth_start > attack_eps:
        raise ValueError('attack_eps_growth_start should be smaller than or equal to rs_eps.')
    if (checkpoint_every is None) ^ (checkpoint_path is None):
        raise ValueError('checkpoint_every and checkpoint_path should be either both None or both not None.')

    if choose_best and val_loader is None:
        raise ValueError('choose_best requires a validation loader')

    validation_tracker = ValidationTracker() if choose_best else None

    # Prepare the epsilon values
    if attack_eps_growth_epoch in [0, 1]:
        epoch_attack_epsilons = [attack_eps] * max_epochs
    else:
        # With num=1, the only value is the initial value (instead of the final one)
        epoch_attack_epsilons = list(np.linspace(attack_eps_growth_start, attack_eps, num=attack_eps_growth_epoch))
        epoch_attack_epsilons += list([attack_eps] * (max_epochs - attack_eps_growth_epoch))
        assert len(epoch_attack_epsilons) == max_epochs

    model.train()
    model.to(device)
    iterator = tqdm(range(max_epochs), desc='Training')

    if loaded_checkpoint is None:
        start_epoch = 0
    else:
        # Epochs are stored internally using 0-indexing
        # Start from the following epoch
        start_epoch = loaded_checkpoint['epoch'] + 1
        model.load_state_dict(loaded_checkpoint['model'])
        optimiser.load_state_dict(loaded_checkpoint['optimiser'])

        if (early_stopping is None) ^ (loaded_checkpoint['early_stopping'] is None):
            raise RuntimeError('There is a mismatch between the current early_stopping and '
                               'the saved one.')

        if early_stopping is not None and loaded_checkpoint['early_stopping'] is not None:
            early_stopping.load_state_dict(loaded_checkpoint['early_stopping'])

        if (validation_tracker is None) ^ (loaded_checkpoint['validation_tracker'] is None):
            raise RuntimeError('There is a mismatch between the current validation_tracker and '
                               'the saved one.')

        if validation_tracker is not None and loaded_checkpoint['validation_tracker'] is not None:
            validation_tracker.load_state_dict(loaded_checkpoint['validation_tracker'])

    input_shape = None

    for epoch in iterator:
        if epoch < start_epoch:
            # Skip previous epochs (happens when loading an existing checkpoint)
            print(f'Skipping epoch {epoch + 1}')
            continue

        current_attack_eps = epoch_attack_epsilons[epoch]
        print(f'Epoch {epoch + 1}')

        # Training phase
        for x, target in train_loader:
            # Move to the correct device
            x = x.to(device)
            target = target.to(device)

            if input_shape is None:
                input_shape = x.shape[1:]

            # Adversarial training: replace some genuine samples with adversarials
            if attack is None:
                x_adv = x
            else:
                #print(f'Adversarial with eps {current_attack_eps}')
                x_adv = adversarial_training(x, target, model, attack, attack_ratio, current_attack_eps)

            # Compute the outputs
            y_pred = model(x_adv)

            # Compute the standard (or adversarial) loss
            # Note: The loss isn't divided by the batch size, although some losses
            # (such as CrossEntropy with mean reduction) do it anyway
            loss = loss_function(y_pred, target)

            # Add the L1 loss
            if l1_regularization != 0:
                loss += l1_loss(model, input_shape, device, l1_regularization)

            optimiser.zero_grad()
            loss.backward()

            # RS Regularization uses a high amount of GPU memory, so we use .backward()
            # for each minibatch. Since .backward() accumulates gradients, this is equivalent
            # to summing all losses and calling .backward() once
            # Note: unlike adversarial eps, rs_eps does not grow with the number of epochs
            if rs_regularization != 0 and (epoch + 1) >= rs_start_epoch:
                #print(f'RS with eps {rs_eps}')
                if rs_minibatch_size is None:
                    rs = rs_loss(model, x, epsilon=rs_eps) * rs_regularization
                    rs.backward()
                else:
                    for minibatch in torch_utils.split_batch(x, rs_minibatch_size):
                        # Note: RS loss isn't divided by the batch size
                        rs = rs_loss(model, minibatch,
                                     epsilon=rs_eps) * rs_regularization
                        rs.backward()

            # Update the weights
            optimiser.step()

            # As a safety measure, remove accumulated gradients
            optimiser.zero_grad()

        # Validation phase
        if val_loader is not None:
            logger.debug('Computing validation loss.')
            val_loss = 0

            for x_val, target_val in val_loader:
                x_val = x_val.to(device)
                target_val = target_val.to(device)

                if attack is None:
                    x_adv_val = x_val
                else:
                    x_adv_val = adversarial_training(x_val, target_val, model, attack, attack_ratio, current_attack_eps)

                with torch.no_grad():
                    y_pred_val = model(x_adv_val)
                    val_loss += loss_function(y_pred_val, target_val)

                    if l1_regularization != 0:
                        assert input_shape is not None
                        val_loss += l1_loss(model, input_shape, device, l1_regularization)

                    if rs_regularization != 0 and (epoch + 1) >= rs_start_epoch:
                        if rs_minibatch_size is None:
                            rs = rs_loss(model, x_val, epsilon=rs_eps) * rs_regularization
                            val_loss += rs
                        else:
                            for minibatch_val in torch_utils.split_batch(x_val, rs_minibatch_size):
                                rs = rs_loss(model, minibatch_val,
                                                epsilon=rs_eps) * rs_regularization
                                val_loss += rs

            iterator.set_description('Training | Validation Loss: {:.3e}'.format(
                val_loss.cpu().detach().item()))

            if validation_tracker is not None:
                validation_tracker(val_loss, model)

            if early_stopping is not None:
                print('Checking early stopping. Loss: ', val_loss)
                early_stopping(val_loss)

                if early_stopping.stop:
                    print('Early stop triggered')
                    logger.debug('Early stop triggered.')
                    break

        if (checkpoint_path is not None) and (epoch + 1) % checkpoint_every == 0:
            if not pathlib.Path(checkpoint_path).exists():
                pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

            # Note: We use 1-indexing for epochs
            current_epoch_path = pathlib.Path(checkpoint_path) / f'{epoch + 1}.check'
            print(f'Saving checkpoint to {current_epoch_path}')
            torch.save({
                'optimiser' : optimiser.state_dict(),
                'epoch' : epoch, # Epochs are stored internally using 0-indexing
                'model' : model.state_dict(),
                'early_stopping' : None if early_stopping is None else early_stopping.state_dict(),
                'validation_tracker' : None if validation_tracker is None else validation_tracker.state_dict()
            }, current_epoch_path)

    if validation_tracker is not None:
        print('Loading best state dict')
        logger.debug('Validation tracker: Loading best state dict.')
        model.load_state_dict(validation_tracker.best_state_dict)

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
            if idx.start < 0 or idx.stop < 0 or idx.step < 0:
                raise NotImplementedError('Negative slices are not supported.')
            slice_ = slice(self.start + idx.start,
                           self.start + idx.stop, idx.step)
            return self.dataset[slice_]
        else:
            if isinstance(idx, int):
                if idx >= self.stop:
                    raise ValueError('Index out of bounds.')
                if idx < 0:
                    raise NotImplementedError('Negative indices are not supported.')
            # Performing checks on tensors could trigger CUDA synchronizations,
            # which would slow down massively the execution
            return self.dataset[self.start + idx]

    def __len__(self):
        return self.stop - self.start

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        assert len(indices) <= len(dataset)
        assert all(i >= 0 for i in indices)
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

# Note: ValidationTracker stores the best overall state_dict without
# considering delta. In other words, even if a nonzero loss improvement is too
# small to reset EarlyStopping's counter, it will be big enough to be
# registered by ValidationTracker.

class ValidationTracker:
    def __init__(self):
        self.best_loss = None
        self.best_state_dict = None

    def __call__(self, val_loss, model):
        val_loss = val_loss.cpu().detach().item()

        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict()
        print('Tracking best loss: ', self.best_loss)

    def state_dict(self):
        return {
            'best_loss' : self.best_loss,
            'best_state_dict' : self.best_state_dict
        }

    def load_state_dict(self, state_dict):
        self.best_loss = state_dict['best_loss']
        self.best_state_dict = state_dict['best_state_dict'] 

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Follows the same semantics as Keras' implementation.
    """

    def __init__(self, patience, delta=0, allow_different_config=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            allow_different_config (bool) : If True, loading a state_dict with different values
                                            for patience and delta will not throw an error.
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.stop = False
        self.delta = delta
        self.allow_different_config = allow_different_config

    def __call__(self, val_loss):
        val_loss = val_loss.cpu().detach().item()

        print('Pre-counter: ', self.counter)
        if self.best_loss is None:
            # First call
            self.best_loss = val_loss
            assert self.counter == 0
        elif val_loss < self.best_loss - self.delta:
            # Significant improvement, reset the counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # Not a significant improvement, increase the counter
            self.counter += 1
            if self.counter >= self.patience:
                # Too many calls without improvement, stop
                self.stop = True
        print('Best loss: ', self.best_loss)
        print('Counter: ', self.counter)

    def state_dict(self):
        return {
            'patience' : self.patience,
            'counter' : self.counter,
            'best_loss' : self.best_loss,
            'stop' : self.stop,
            'delta' : self.delta
        }

    def load_state_dict(self, state_dict):
        if self.patience != state_dict['patience']:
            if self.allow_different_config:
                logger.warning('Loading a different value for patience.')
            else:
                raise RuntimeError('Found a different value for patience. If this is '
                                   'intentional, initialise with allow_different_config=True.')

        if self.delta != state_dict['delta']:
            if self.allow_different_config:
                logger.warning('Loading a different value for delta.')
            else:
                raise RuntimeError('Found a different value for delta. If this is '
                                   'intentional, initialise with allow_different_config=True.')

        self.patience = state_dict['patience']
        self.counter = state_dict['counter']
        self.best_loss = state_dict['best_loss']
        self.stop = state_dict['stop']
        self.delta = state_dict['delta']
