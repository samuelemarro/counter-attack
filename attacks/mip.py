import logging

import advertorch
import numpy as np
import torch
import torch.nn as nn

import torch_utils
import utils

logger = logging.getLogger(__name__)

# Required libraries:
# PyCall (installed from the Python package julia) (alternatively, I can add it as its own library)
# JuMP
# MIPVerify


# TODO: Rimuovere
mipverify_path = 'C:/Users/Samuele/source/MIPVerify.jl/src/MIPVerify.jl'
from julia import Main
Main.include(mipverify_path)

# TODO: Integrarli nella classe?
def unravel_sequential(sequential):
    layers = []

    def recursive_unravel(module):
        if isinstance(module, nn.Sequential):
            for submodule in module:
                recursive_unravel(submodule)
        else:
            layers.append(module_to_mip(module))

    recursive_unravel(sequential)

    return layers

def convert_sequential(sequential, reference_image):
    from julia import MIPVerify

    layers = unravel_sequential(sequential)
    converted_layers = []

    x = reference_image
    for layer in layers:
        pre_shape = x.shape
        x = layer(x)
        post_shape = x.shape

        converted_layers.append(convert_module(layer, pre_shape, post_shape))

    return MIPVerify.Sequential(converted_layers)

def module_to_mip(module):
    # TODO: Nella versione finale, MIPVerify esiste
    #from julia import MIPVerify
    from julia import Main
    MIPVerify = Main.MIPVerify

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    if isinstance(module, nn.Linear):
        converted = MIPVerify.Linear(
            to_numpy(module.weight).transpose(),
            to_numpy(module.bias)
        )
    elif isinstance(module, nn.Flatten):
        assert module.end_dim == -1
        # Il parametro da passare è essenzialmente il numero di dimensioni dell'input
        # Interpreta ndim=k come se prima dovesse fare una trasposizione
        # [k-1, k-2, ..., 0] (inverte gli assi) e poi un flatten verso un vettore 1D
        # Perché fa la trasposizione? Forse perché Julia è column-major?

        #### converted = MIPVerify.Flatten(2) # TODO: Con che parametri?
        # TODO: Riportare all'originale
        # Per CIFAR: [1, 2, 3, 4]
        # Per MNIST: [1, 2]? O [2, 1]?
        # Forse deve essere invertito per la questione del column-major! Sì!
        converted = MIPVerify.Flatten(4, [4, 3, 2, 1])
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
    elif isinstance(module, nn.Conv2d):
        # PyTorch follows the filter format:
        # (out_channels, in_channels / groups, kernel_size[0], kernel_size[1])
        # MIP expects the following format:
        # (filter_height, filter_width, in_channels, out_channels)
        # Remember: PyTorch is height-first
        # MIP uses "same" padding and a (1, 1, 1, 1) stride

        # TODO: Handle uneven padding
        # Con le modifiche potrei ora gestire ogni tipo di padding

        if not all(x == 1 for x in module.dilation):
            raise ValueError('MIP only supports (1, 1)-style dilation. Received: {}.'.format(module.dilation))
        if not all(x == 0 for x in module.output_padding):
            raise ValueError('MIP does not support output padding. Received: {}.'.format(module.output_padding))
        if module.groups != 1:
            raise ValueError('MIP only supports convolutions with groups=1. Received: {}'.format(module.groups))
        if len(set(module.stride)) > 1:
            raise ValueError('MIP only supports striding along all dimensions with the same value. Received: {}'.format(module.stride))

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
        else:
            stride = module.stride

        filter_ = module.weight.cpu().detach().numpy()
        # Transpose the filter to match MIPVerify
        # TODO: Attenzione alla posizione del canale!
        filter_ = np.transpose(filter_, [2, 3, 1, 0])

        padding = module.padding

        bias = module.bias.cpu().detach().numpy()

        converted = MIPVerify.Conv2d(filter_, bias, stride, padding)

        # Simulate MIPVerify's padding and compare it with the actual one
        # TODO: Richiede la dimensione dell'input, che io non ho
        """in_height = stride.filter
        out_height = ceil(Int, in_height/stride)
        out_width = ceil(Int, in_width/stride)
        out_height = 
        pad_along_height = max((out_height - 1)*stride + filter_height - in_height, 0)
        pad_along_width = max((out_width - 1)*stride + filter_width - in_width, 0)
        filter_height_offset = round(Int, pad_along_height/2, RoundDown)
        filter_width_offset = round(Int, pad_along_width/2, RoundDown)"""

        # L'immagine (H, W) filtrata (senza padding) dal filtro (K_H, K_W)
        # avrà dimensione (H - K_H + 1, W - K_W + 1)

        # TODO: Testare che sia corretto
        # TODO: Implementare batch_norm (vedi foglio) e max_pooling
    elif isinstance(module, nn.MaxPool2d):
        if module.padding != 0:
            raise ValueError('MIPVerify does not support padding for MaxPool. Received: "{}".'.format(module.padding))

        if module.dilation != 1:
            raise ValueError('MIPVerify does not support dilation for MaxPool. Received: "{}".'.format(module.dilation))

        if module.stride != module.kernel_size:
            raise ValueError('MIPVerify does not support MaxPool with stride != kernel_size. Received: "{}", "{}".'.format(module.stride, module.dilation))

        if module.return_indices:
            raise ValueError('MIPVerify does not support return_indices=True.')
        if isinstance(module.stride, tuple):
            stride = module.stride
        else:
            # TODO: Perché?
            stride = (1, module.stride, module.stride, 1)

        converted = MIPVerify.MaxPool(stride)
    elif isinstance(module, nn.BatchNorm2d):
        # Batch Normalization computes the formula:
        # y = (x - rmean) * gamma / sqrt(rvar + eps) + beta
        # where rmean is the running mean, rvar is the running variance,
        # eps is a small value and gamma and beta are optional parameters
        # If gamma and beta are not used (i.e. affine=False), the default value
        # of beta is 0 and the default value of gamma is 1
        #
        # MIPVerify's normalization uses the formula:
        # y = (x - mean) / std
        # We can therefore convert batch normalization to MIPVerify's
        # normalization using
        # mean = rmean - beta * sqrt(rvar + eps) / gamma
        # std = sqrt(rvar + eps) / gamma

        if module.weight is None:
            gamma = 1
        else:
            gamma = module.weight.detach().cpu().numpy()

        if module.bias is None:
            beta = 0
        else:
            beta = module.bias.detach().cpu().numpy()

        running_mean = module.running_mean.detach().cpu().numpy()
        running_var = module.running_var.detach().cpu().numpy()
        eps = module.eps

        mean = running_mean - beta * np.sqrt(running_var + eps) / gamma
        std = np.sqrt(running_var + eps) / gamma

        converted = MIPVerify.Normalize(mean, std)
    else:
        raise NotImplementedError('Unsupported module "{}".'.format(type(module).__name__)) # TODO: Supporto altri moduli?

    return converted

def sequential_to_mip(sequential):
    # TODO: Nella versione finale, MIPVerify esiste
    #from julia import MIPVerify
    from julia import Main
    MIPVerify = Main.MIPVerify

    converted_layers = []

    def recursive_parser(module):
        if isinstance(module, nn.Sequential):
            for submodule in module:
                recursive_parser(submodule)
        else:
            converted_layers.append(module_to_mip(module))

    recursive_parser(sequential)

    return MIPVerify.Sequential(converted_layers, 'Converted network')

class MIPAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    _pyjulia_installed = False

    def __init__(self, predict, p, targeted, tolerance=1e-6, clip_min=0, clip_max=1, solver='gurobi'):
        super().__init__(predict, None, clip_min, clip_max)
        if p in [1, 2, float('inf')]:
            self.p = p
        else:
            raise NotImplementedError('MIPAttack only supports p=1, 2 or inf.')

        # Lazy import
        if not MIPAttack._pyjulia_installed:
            import julia
            julia.install()
            _pyjulia_installed = True

        self.mip_model = sequential_to_mip(predict)
        self.targeted = targeted
        self.tolerance = tolerance

        if solver == 'gurobi':
            from julia import Gurobi
            # TODO: L'opzione extra è aggiunta per ridurre l'OutOfMemory
            self.solver = Gurobi.GurobiSolver(OutputFlag=1)
        elif solver == 'cbc':
            from julia import Cbc
            main_solver = Cbc.CbcSolver(logLevel=0)
            tightening_solver = Cbc.CbcSolver(logLevel=0, seconds=20)
            # TODO: Come gestire il tightening e main?
        else:
            raise NotImplementedError('Unsupported solver "{}".'.format(solver))

    def mip_attack(self, image, label):
        # TODO: Nella versione finale, MIPVerify esiste
        #from julia import MIPVerify
        from julia import Main
        MIPVerify = Main.MIPVerify

        from julia import JuMP
        # TODO: Install jump?

        # Julia is 1-indexed
        target_label = label + 1

        # TODO: è necessario eseguire modifiche quando si lavora in CIFAR?
        # TODO: Torch is channel-first, what is MIP?
        image = image.transpose([1, 2, 0])
        
        extra_dimension = image.shape[-1] == 1

        #if extra_dimension:
        #    image = image.squeeze(axis=-1)

        # TODO: Test
        image = np.expand_dims(image, 0)

        # TODO: Verificare invert
        adversarial_result = MIPVerify.find_adversarial_example(self.mip_model,
                                    image, target_label, self.solver, norm_order=self.p,
                                    tolerance=self.tolerance, invert_target_selection=not self.targeted)

        adversarial = np.array(JuMP.getvalue(adversarial_result['PerturbedInput']))

        #if extra_dimension:
        #    adversarial = np.expand_dims(adversarial, -1)

        # Reconvert to channel-first
        adversarial = adversarial.transpose([0, 3, 1, 2])

        # Remove the batch dimension
        adversarial = adversarial.squeeze(axis=0)

        return adversarial

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        adversarials = []

        for image, label in zip(x, y):
            image = image.detach().cpu().numpy()
            label = label.detach().cpu().numpy().item()
            
            adversarial = self.mip_attack(image, label)
            adversarials.append(torch.from_numpy(adversarial).to(x))

        return utils.maybe_stack(adversarials, x.shape[1:], device=x.device)