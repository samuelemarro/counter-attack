import torch
import numpy as np

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

class BatchLimitedModel(torch.nn.Module):
    def __init__(self, wrapped_model, batch_size):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.batch_size = batch_size

    def forward(self, x):
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))

        outputs = []

        for batch_id in range(nb_batches):
            batch_begin = batch_id * self.batch_size
            batch_end = (batch_id + 1) * self.batch_size

            batch = x[batch_begin:batch_end]
            
            outputs.append(self.wrapped_model(batch))

        outputs = torch.cat(outputs)

        assert len(outputs) == len(x)

        return outputs

class Normalisation(torch.nn.Module):
    def __init__(self, means, stdevs):
        super().__init__()
        self.means = torch.from_numpy(np.array(means).reshape((3, 1, 1)))
        self.stdevs = torch.from_numpy(np.array(stdevs).reshape((3, 1, 1)))

    def forward(self, input):
        means = self.means.to(input)
        stdevs = self.stdevs.to(input)
        return (input - means) / stdevs

# Versione modulare di torch.squeeze()
class Squeeze(torch.nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, self.dim)

def train(model, train_loader, optimiser, loss, max_epochs, val_loader=None, additional_metrics={}):
    metrics = additional_metrics
    if 'Loss' not in metrics.keys():
        metrics['Loss'] = Loss(loss)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    trainer = create_supervised_trainer(model, optimiser, loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics=metrics,
                                            device=device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        output_string = 'Training Results - Epoch {}: '.format(trainer.state.epoch)
        for key, value in evaluator.state.metrics.items():
            output_string += '\n{}: {}'.format(key, value)
        print(output_string)

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(val_loader)
            output_string = 'Validation Results - Epoch {}: '.format(trainer.state.epoch)
            for key, value in evaluator.state.metrics.items():
                output_string += '\n{}: {}'.format(key, value)
            print(output_string)

    trainer.run(train_loader, max_epochs=max_epochs)