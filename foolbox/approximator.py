import torch
import ignite

from torch import nn
from torch.optim import SGD
import torch.nn.functional as F

from tqdm import tqdm

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

import cifar10_models

# Nota: Puoi ottenere un approximator creando
# un classifier con una classe e addestrando in MSE

def get_approximator():
    return cifar10_models.resnet50(num_classes=1)

def train_approximator(model, images, distances, epochs, lr, momentum, log_interval=1):
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(images), torch.from_numpy(distances))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)

    model.train()

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    
    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, distances = data

            inputs = inputs.to(device)
            distances = distances.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, distances)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_interval == 0:    # print every 2000 mini-batches
                print('[{}, {}] loss: {:.2e}'.format(
                    epoch + 1, i + 1, running_loss / log_interval))
                running_loss = 0.0



"""def train_approximator(model, images, distances, epochs, lr, momentum, log_interval=1):
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(images), torch.from_numpy(distances))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = create_supervised_trainer(model, optimizer, F.mse_loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'mse': Loss(F.mse_loss)},
                                            device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_mse = metrics['mse']
        tqdm.write(
            "Training Results - Epoch: {}  Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_mse)
        )
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()"""