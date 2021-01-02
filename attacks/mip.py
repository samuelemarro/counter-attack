import logging
import time

import advertorch
import numpy as np
import torch
from torch import per_tensor_affine
import torch.nn as nn

import torch_utils
import utils

logger = logging.getLogger(__name__)

# Required libraries:
# PyCall (installed from the Python package "julia") (alternatively, I can add it as its own library)
# JuMP
# MIPVerify

# TODO: Testare MaxPool e BatchNorm

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
        # Il parametro da passare è essenzialmente il numero di dimensioni dell'input
        # Interpreta ndim=k come se prima dovesse fare una trasposizione
        # [k-1, k-2, ..., 0] (inverte gli assi) e poi un flatten verso un vettore 1D
        # Perché fa la trasposizione? Forse perché Julia è column-major?

        # PyTorch uses the BCHW format and reads row-first (backwards), so the order is WHCB
        # Julia uses the BHWC format and reads column-first (straightforward), so in order to read it
        # in the same way as PyTorch, we have to permute the dimensions to WHCB using the permutation:
        # [3, 2, 4, 1]
        converted = MIPVerify.Flatten(4, [3, 2, 4, 1])
    elif isinstance(module, nn.ReLU):
        converted = MIPVerify.ReLU()
    elif isinstance(module, torch_utils.MaskedReLU):
        always_zero = module.always_zero.data
        always_linear = module.always_linear.data
        print(torch.sum(always_zero.float()))
        assert not (always_zero & always_linear).any()
        mask = torch.zeros_like(always_zero, dtype=float)

        mask += always_linear.float()
        mask -= always_zero.float()

        converted = MIPVerify.MaskedReLU(mask)
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

        # L'immagine (H, W) filtrata (senza padding) dal filtro (K_H, K_W)
        # avrà dimensione (H - K_H + 1, W - K_W + 1)

        # TODO: Testare che sia corretto
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

        if module.training:
            raise ValueError('Batch normalization must be in eval mode.')

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
    from julia import MIPVerify

    layers = torch_utils.unpack_sequential(sequential)

    layers = [module_to_mip(layer) for layer in layers]

    return MIPVerify.Sequential(layers, 'Converted network')

# TODO: Return None if the solver times out

# TODO: Retry system (restart n times with higher correction factor)
# Il test su x9 falliva con correction_factor=1, ricontrollare ora che è 1.25

class MIPAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    _pyjulia_installed = False

    def __init__(self, predict, p, targeted, tolerance=1e-6, clip_min=0, clip_max=1, correction_factor=1.25, tightening_overrides=dict(), **gurobi_kwargs):
        super().__init__(predict, None, clip_min, clip_max)
        if p in [1, 2, float('inf')]:
            self.p = p
        else:
            raise NotImplementedError('MIPAttack only supports p=1, 2 or inf.')

        # Lazy import
        if not MIPAttack._pyjulia_installed:
            import julia
            julia.install()

        self.mip_model = sequential_to_mip(predict)
        self.targeted = targeted
        self.correction_factor = correction_factor

        if tolerance == 0:
            logger.warn('MIP\'s tolerance is set to 0. Given the possible numerical errors,'
            ' it is likely that MIP\'s adversarials will be considered unsuccessful by Torch\'s'
            ' model.')

        self.tolerance = tolerance

        # To avoid modifying the default value
        tightening_overrides = dict(tightening_overrides)

        if 'TimeLimit' in gurobi_kwargs and gurobi_kwargs['TimeLimit'] == 0:
            gurobi_kwargs['TimeLimit'] = np.inf

        if 'TimeLimit' in tightening_overrides and tightening_overrides['TimeLimit'] == 0:
            tightening_overrides['TimeLimit'] = np.inf

        tightening_kwargs = dict(gurobi_kwargs)

        for key, value in tightening_overrides.items():
            tightening_kwargs[key] = value
        
        from julia import Gurobi
        self.solver = Gurobi.GurobiSolver(OutputFlag=1, **gurobi_kwargs)
        # TODO: Capire bene come funziona Gurobi.Env()
        self.tightening_solver = Gurobi.GurobiSolver(Gurobi.Env(),
                                                    OutputFlag=0,
                                                    **tightening_kwargs)

    def mip_attack(self, image, label, starting_point=None):
        from julia import MIPVerify

        from julia import JuMP
        # TODO: Install jump?

        start_time = time.clock()

        # Julia is 1-indexed
        target_label = label + 1

        if starting_point is None:
            perturbation = MIPVerify.UnrestrictedPerturbationFamily()
        else:
            if not np.isposinf(self.p):
                raise NotImplementedError('Starting point is only supported for the Linf norm.')

            pre_distance = np.linalg.norm((image - starting_point).flatten(), ord=np.inf).item()
            pre_distance *= self.correction_factor
            perturbation = MIPVerify.LInfNormBoundedPerturbationFamily(pre_distance)

            starting_point = starting_point.transpose([1, 2, 0])
            starting_point = np.expand_dims(starting_point, 0)

        
        image = image.transpose([1, 2, 0])
        image = np.expand_dims(image, 0)
        
        adversarial_result = MIPVerify.find_adversarial_example(self.mip_model,
                                    image, target_label, self.solver, norm_order=self.p,
                                    tolerance=self.tolerance, invert_target_selection=not self.targeted,
                                    tightening_solver=self.tightening_solver, pp=perturbation)

        adversarial = np.array(JuMP.getvalue(adversarial_result['PerturbedInput']))

        elapsed_time = time.clock() - start_time
        adversarial_result['WallClockTime'] = elapsed_time

        #if extra_dimension:
        #    adversarial = np.expand_dims(adversarial, -1)

        # Reconvert to channel-first
        adversarial = adversarial.transpose([0, 3, 1, 2])

        # Remove the batch dimension
        adversarial = adversarial.squeeze(axis=0)
        return adversarial, adversarial_result

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        adversarials = []

        for image, label in zip(x, y):
            image = image.detach().cpu().numpy()
            label = label.detach().cpu().numpy().item()
            
            adversarial, _ = self.mip_attack(image, label)

            if np.any(np.isnan(adversarial)):
                adversarial = None
            else:
                adversarial = torch.from_numpy(adversarial).to(x)

            adversarials.append(adversarial)

        # TODO: Ritornare semplicemente adversarials
        return utils.maybe_stack(adversarials, x.shape[1:], device=x.device)

    def perturb_advanced(self, x, y=None, starting_points=None):
        x, y = self._verify_and_process_inputs(x, y)

        adversarials = []
        lower_bounds = []
        upper_bounds = []
        solve_times = []

        if starting_points is None:
            starting_points = [None] * len(x)

        for image, label, starting_point in zip(x, y, starting_points):
            image = image.detach().cpu().numpy()
            label = label.detach().cpu().numpy().item()

            if starting_point is not None:
                starting_point = starting_point.detach().cpu().numpy()

            adversarial, adversarial_result = self.mip_attack(image, label, starting_point)
            if np.any(np.isnan(adversarial)):
                adversarial = None
            else:
                adversarial = torch.from_numpy(adversarial).to(x)

            from julia import JuMP

            lower = JuMP.getobjectivebound(adversarial_result['Model'])
            if np.isnan(lower):
                lower = None

            upper = JuMP.getobjectivevalue(adversarial_result['Model'])
            if np.isnan(upper):
                upper = None

            solve_time = adversarial_result['WallClockTime']

            adversarials.append(adversarial)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            solve_times.append(solve_time)

        return adversarials, lower_bounds, upper_bounds, solve_times