import logging

import advertorch
import numpy as np
import torch
import torch.nn as nn

import torch_utils
import utils

logger = logging.getLogger(__name__)

def module_to_mip(module):
    from julia import MIPVerify

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    if isinstance(module, nn.Linear):
        converted = MIPVerify.Linear(
            to_numpy(module.weight).transpose(),
            to_numpy(module.bias)
        )
    elif isinstance(module, nn.Flatten):
        assert module.end_dim == -1
        converted = MIPVerify.Flatten(2) # TODO: Con che parametri?
    elif isinstance(module, nn.ReLU):
        converted = MIPVerify.ReLU()
    elif isinstance(module, torch_utils.Normalisation):
        mean = np.squeeze(to_numpy(module.mean))
        std = np.squeeze(to_numpy(module.std))
        
        # TODO: Capire qual è la forma adeguata
        if len(mean.shape) == 0:
            mean = [mean.item()]
        if len(std.shape) == 0:
            std = [std.item()]

        converted = MIPVerify.Normalize(mean, std)
    else:
        raise NotImplementedError('Unsupported module "{}".'.format(type(module).__name__)) # TODO: Supporto altri moduli?

    return converted

def sequential_to_mip(sequential):
    from julia import MIPVerify

    converted_layers = []

    def recursive_parser(module):
        if isinstance(module, nn.Sequential):
            for submodule in module:
                recursive_parser(submodule)
        else:
            converted_layers.append(module_to_mip(module))

    recursive_parser(sequential)

    return MIPVerify.Sequential(converted_layers, 'Converted network')

#TODO: Come gestire il fatto che si ottengono output esatti, che possono quindi coincidere?
# Opzione 1: Mettere tolerance
# Opzione 2: Se la label è la stessa, verificare se la c'è un output molto vicino
# Opzione 3: Se la label è la stessa, riprovare con tolerance > 0

# TODO: E se invert_target non funzionava a causa dei problemi con la tolerance?

class MIPAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, p, num_labels, targeted, tolerance=1e-6, clip_min=0, clip_max=1, solver='gurobi'):
        super().__init__(predict, None, clip_min, clip_max)
        if p in [1, 2, float('inf')]:
            self.p = p
        else:
            raise NotImplementedError('MIPAttack only supports p=1, 2 or inf.')

        # Lazy import
        import julia
        # TODO: Check if it's already installed?
        julia.install()

        self.num_labels = num_labels
        self.mip_model = sequential_to_mip(predict)
        self.targeted = targeted
        self.tolerance = tolerance

        if solver == 'gurobi':
            from julia import Gurobi
            self.solver = Gurobi.GurobiSolver(OutputFlag=0)
        else:
            raise NotImplementedError('Unsupported solver "{}".'.format(solver))

    def mip_attack(self, image, label):
        from julia import MIPVerify
        from julia import JuMP
        # Julia is 1-indexed
        if self.targeted:
            mip_target_label = label + 1
        else:
            target_labels = [x + 1 for x in range(self.num_labels) if x != label]

        # TODO: è necessario eseguire modifiche quando si lavora in CIFAR?
        # TODO: Torch is channel-first, what is MIP?
        image = image.transpose([1, 2, 0])
        
        extra_dimension = image.shape[-1] == 1

        if extra_dimension:
            image = image.squeeze(axis=-1)

        adversarial_result = MIPVerify.find_adversarial_example(self.mip_model,
                                    image, target_labels, self.solver, norm_order=self.p,
                                    tolerance=self.tolerance)

        adversarial = np.array(JuMP.getvalue(adversarial_result['PerturbedInput']))

        if extra_dimension:
            adversarial = np.expand_dims(adversarial, -1)

        # Reconvert to CWH
        adversarial = adversarial.transpose([2, 0, 1])

        return adversarial

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        adversarials = []

        for image, label in zip(x, y):
            print(image.dtype)
            image = image.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            print(image.dtype)
            print(label.dtype)
            
            adversarial = self.mip_attack(image, label)
            adversarials.append(torch.from_numpy(adversarial).to(x))

        return utils.maybe_stack(adversarials, x.shape[1:], device=x.device)