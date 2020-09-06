import torch
import numpy as np

from tqdm import tqdm

class BatchLimitedModel(torch.nn.Module):
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

class Normalisation(torch.nn.Module):
    def __init__(self, mean, std, num_channels=3):
        super().__init__()
        self.mean = torch.from_numpy(np.array(mean).reshape((num_channels, 1, 1)))
        self.std = torch.from_numpy(np.array(std).reshape((num_channels, 1, 1)))

    def forward(self, input):
        mean = self.mean.to(input)
        std = self.std.to(input)
        return (input - mean) / std

# Modular version of torch.squeeze()
class Squeeze(torch.nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, self.dim)

def train(model, train_loader, optimiser, loss_function, max_epochs, device, val_loader=None, l1_regularization=0):
    model.to(device)

    for i in tqdm(range(max_epochs), desc='Training'):
        for x, target in train_loader:
            x = x.to(device)
            target = target.to(device)

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
                    target_val = target_val.to(target_val)

                    y_pred_val = model(x_val)
                    val_loss += loss_function(y_pred_val, target_val)

            print('Validation Loss: {:.3e}'.format(val_loss.cpu().detach().item()))


class FirstNDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_samples):
        if num_samples < 1:
            raise ValueError('num_samples must be at least 1.')

        self.dataset = dataset
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.num_samples