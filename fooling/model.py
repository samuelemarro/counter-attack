import sys

sys.path.append('.')

import json
import attacks

from fooling.estimators import NESWrapper
from fooling.nes import NESAttack


import advertorch.attacks
import click
import torch
import torch.nn as nn

import parsing
import utils

class DistanceModel(nn.Module):
    def __init__(self, predict, attack_fn, nb_samples, fd_eta=1e-3):
        super().__init__()
        self.predict = predict
        self.attack_fn = attack_fn

        def distance_fn(x):
            x = x.reshape(x.shape[0], 1, 28, 28)
            with torch.enable_grad():
                adversarial = self.attack_fn(x)
            
            distance = torch.abs(adversarial - x)
            distance, _ = torch.max(distance, dim=3)
            distance, _ = torch.max(distance, dim=2)
            distance, _ = torch.max(distance, dim=1)
            # TODO: Treat is as a class
            return distance.reshape(x.shape[0], 1)
        
        self.distance_estimator = NESWrapper(distance_fn, nb_samples, fd_eta)

    def forward(self, x):
        output = self.predict(x)

        flattened_x = torch.flatten(x, start_dim=1)
        distance = self.distance_estimator(flattened_x)
        return output, distance

class CustomLossFunction(nn.Module):
    def __init__(self, loss_fn, coeff):
        super().__init__()
        self.loss_fn = loss_fn
        self.coeff = coeff
    def forward(self, outputs, targets):
        real_outputs, distance = outputs
        real_loss = self.loss_fn(real_outputs, targets)

        return real_loss + self.coeff * distance

#@click.command()
#@click.argument('domain')
#@click.argument('architecture')
def main(domain, architecture):
    loss_function = CustomLossFunction(nn.CrossEntropyLoss(), 1)
    # Why is normalization = False?
    model = parsing.parse_model(domain, architecture, f'trained-models/classifiers/standard/{domain}-{architecture}.pth', False, False, False, True)
    model.eval()

    dataset = parsing.parse_dataset(domain, 'std:test')
    attack_config = utils.read_attack_config_file('balanced_attack_configuration.cfg')

    counter_attack = parsing.parse_attack_pool(['fast_gradient', 'uniform'], domain, float('inf'), 'standard', model, attack_config, 'cuda')

    with open(f'analysis/distances/balanced/{domain}-{architecture}-standard.json', 'r') as f:
        distances = json.load(f)

    wrapped_model = DistanceModel(model, counter_attack, 200)
    wrapped_model.to('cuda')
    wrapped_model.train()

    class InnerAttack():
        def __init__(self, predict, loss_fn, clip_min=0, clip_max=1) -> None:
            self.predict = predict
            self.loss_fn = loss_fn
            self.clip_min = clip_min
            self.clip_max = clip_max
            self.targeted = False
        def __call__(self, x, y, eps):
            a = advertorch.attacks.LinfPGDAttack(self.predict, loss_fn=self.loss_fn, eps=eps, nb_iter=1, eps_iter=0.01, rand_init=True, clip_min=0, clip_max=1)
            return a.perturb(x, y)
    
    #main_attack = attacks.EpsilonBinarySearchAttack(InnerAttack(wrapped_model, loss_function), float('inf'))

    index = int(list(distances.keys())[0])

    genuine = torch.unsqueeze(dataset[index][0], dim=0).to('cuda')
    label = torch.unsqueeze(model(torch.unsqueeze(genuine, dim=0)).argmax(), dim=0).to('cuda')

    print(genuine.shape)
    print(label.shape)

    #main_attack.perturb(genuine, label)
    main_attack = InnerAttack(wrapped_model, loss_function)
    adversarial = main_attack(genuine, label, 0.1)


    

if __name__ == '__main__':
    main('mnist', 'a')