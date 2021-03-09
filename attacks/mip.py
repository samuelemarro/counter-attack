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
# PyCall (installed from the Python package "julia")
# JuMP
# MIPVerify

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

        # PyTorch uses the BCHW format and reads row-first (backwards), so the order is WHCB
        # Julia uses the BHWC format and reads column-first (straightforward), so in order to read it
        # in the same way as PyTorch, we have to permute the dimensions to WHCB using the permutation:
        # [3, 2, 4, 1]
        converted = MIPVerify.Flatten(4, [3, 2, 4, 1])
    elif isinstance(module, nn.ReLU):
        converted = MIPVerify.ReLU()
    elif isinstance(module, torch_utils.MaskedReLU):
        always_zero = module.always_zero.data.cpu().numpy()
        always_linear = module.always_linear.data.cpu().numpy()
        assert not (np.logical_and(always_zero, always_linear)).any()
        assert always_zero.shape == always_linear.shape

        always_zero = always_zero.astype(np.float)
        always_linear = always_linear.astype(np.float)

        logger.debug('Adding %s always_zero elements and %s always_linear elements (out of %s elements).',
            np.sum(always_zero), np.sum(always_linear), np.prod(always_zero.shape))

        mask = np.zeros_like(always_zero)

        mask += always_linear
        mask -= always_zero

        mask = np.expand_dims(mask.transpose([1, 2, 0]), 0)

        converted = MIPVerify.MaskedReLU(mask)
    elif isinstance(module, torch_utils.Normalisation):
        mean = np.squeeze(to_numpy(module.mean))
        std = np.squeeze(to_numpy(module.std))

        assert mean.shape == std.shape == ()

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

        if not all(x == 1 for x in module.dilation):
            raise ValueError(
                f'MIP only supports (1, 1)-style dilation. Received: {module.dilation}.')
        if not all(x == 0 for x in module.output_padding):
            raise ValueError(
                f'MIP does not support output padding. Received: {module.output_padding}.')
        if module.groups != 1:
            raise ValueError(
                f'MIP only supports convolutions with groups=1. Received: {module.groups}')
        if len(set(module.stride)) > 1:
            raise ValueError(
                f'MIP only supports striding along all dimensions with the same value. Received: {module.stride}')

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
        else:
            stride = module.stride

        filter_ = module.weight.cpu().detach().numpy()
        # Transpose the filter to match MIPVerify
        filter_ = np.transpose(filter_, [2, 3, 1, 0])

        padding = module.padding

        bias = module.bias.cpu().detach().numpy()

        converted = MIPVerify.Conv2d(filter_, bias, stride, padding)

        # The image (H, W) after a zero padding convolution with filter (K_H, K_W)
        # will have shape (H - K_H + 1, W - K_W + 1)
    else:
        raise NotImplementedError(
            f'Unsupported module "{type(module).__name__}".')

    return converted


def sequential_to_mip(sequential):
    from julia import MIPVerify

    layers = torch_utils.unpack_sequential(sequential)

    layers = [module_to_mip(layer) for layer in layers]

    conversion_time = time.time()

    return MIPVerify.Sequential(layers, f'Converted network ({conversion_time})')

# TODO: Return None if the solver times out

# TODO: Retry system (restart n times with higher correction factor)
# Il test su x9 falliva con correction_factor=1, ricontrollare ora che è 1.25


class MIPAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    _pyjulia_installed = False

    def __init__(self, predict, p, targeted, tolerance=1e-6, clip_min=0,
                clip_max=1, initial_correction_factor=1.05, attempts=3,
                correction_factor_growth=1.5, retry_absolute_gap=1e-5,
                tightening_overrides=None, **gurobi_kwargs):
        super().__init__(predict, None, clip_min, clip_max)

        if tightening_overrides is None:
            tightening_overrides = dict()

        if not np.isposinf(p):
            raise NotImplementedError('MIPAttack only supports p=1, 2 or inf.')

        self.p = p

        # Lazy import
        if not MIPAttack._pyjulia_installed:
            import julia
            julia.install()

        self.mip_model = sequential_to_mip(predict)
        self.targeted = targeted
        self.initial_correction_factor = initial_correction_factor
        self.max_attempts = attempts
        self.correction_factor_growth = correction_factor_growth
        self.retry_absolute_gap = retry_absolute_gap

        if tolerance == 0:
            logger.warning('MIP\'s tolerance is set to 0. Given the possible numerical errors,'
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

        self._checked_model = False

    def _check_model(self, image, threshold=1e-3):
        model_device = next(self.predict.parameters()).device
        torch_output = self.predict(torch.unsqueeze(torch.tensor(image).to(model_device), 0)).detach().cpu().numpy()

        image = image.transpose([1, 2, 0])
        image = np.expand_dims(image, 0)
        julia_output = self.mip_model(image)

        return np.max(np.abs(torch_output - julia_output)) <= threshold

    def perform_attempt(self, image, label, starting_point=None, perturbation_size=None):
        from julia import MIPVerify
        from julia import JuMP

        start_time = time.clock()

        # Julia is 1-indexed
        target_label = label + 1

        if perturbation_size is None:
            logger.debug('No starting point provided, using unbounded perturbation.')
            perturbation = MIPVerify.UnrestrictedPerturbationFamily()
        else:
            if not np.isposinf(self.p):
                raise NotImplementedError(
                    'Perturbation size is only supported for the Linf norm.')
            perturbation = MIPVerify.LInfNormBoundedPerturbationFamily(
                perturbation_size)

        if starting_point is not None:
            starting_point = starting_point.transpose([1, 2, 0])
            starting_point = np.expand_dims(starting_point, 0)

        image = image.transpose([1, 2, 0])
        image = np.expand_dims(image, 0)

        """adversarial_result = MIPVerify.find_adversarial_example(self.mip_model,
                                                                image, target_label, self.solver, norm_order=self.p,
                                                                tolerance=self.tolerance, invert_target_selection=not self.targeted,
                                                                tightening_solver=self.tightening_solver, pp=perturbation)"""
        from julia import Main
        Main.include('mip_interface.jl')

        adversarial_result = Main.find_adversarial_example(self.mip_model,
                                    image, target_label, self.solver, norm_order=self.p,
                                    tolerance=self.tolerance, invert_target_selection=not self.targeted,
                                    tightening_solver=self.tightening_solver, rebuild=False, pp=perturbation,
                                    starting_point=starting_point)

        adversarial = np.array(JuMP.getvalue(
            adversarial_result['PerturbedInput']))

        elapsed_time = time.clock() - start_time
        adversarial_result['WallClockTime'] = elapsed_time

        # if extra_dimension:
        #    adversarial = np.expand_dims(adversarial, -1)

        # Reconvert to channel-first
        adversarial = adversarial.transpose([0, 3, 1, 2])

        # Remove the batch dimension
        adversarial = adversarial.squeeze(axis=0)
        return adversarial, adversarial_result

    def mip_attack(self, image, label, heuristic_starting_point=None):
        if not self._checked_model:
            assert self._check_model(image, threshold=1e-3)
            self._checked_model = True

        from julia import JuMP

        starting_point = heuristic_starting_point
        perturbation_size = None
        upper = np.nan
        lower = 0
        attempts = 0

        # Initialized by heuristic
        if starting_point is not None:
            perturbation_size = np.linalg.norm(
                    (image - starting_point).flatten(), ord=np.inf).item()

            perturbation_size *= self.initial_correction_factor

            while np.isnan(upper) and attempts < self.max_attempts:
                attempts += 1
                logger.debug('Attempt #%s (feasibility).', attempts)
                adversarial, adversarial_result = self.perform_attempt(image, label, starting_point=starting_point, perturbation_size=perturbation_size)
                upper = JuMP.getobjectivevalue(adversarial_result['Model'])
                lower = JuMP.getobjectivebound(adversarial_result['Model'])

                if np.isnan(upper):
                    # MIP failed to find a feasible solution, relax the perturbation size constraint
                    perturbation_size *= self.correction_factor_growth

            if not np.isnan(upper):
                starting_point = adversarial

        while ((upper - lower) > self.retry_absolute_gap or np.isnan(upper)) and attempts < self.max_attempts:
            attempts += 1
            logger.debug('Attempt #%s (optimality)', attempts)
            adversarial, adversarial_result = self.perform_attempt(image, label, starting_point=starting_point, perturbation_size=perturbation_size)
            upper = JuMP.getobjectivevalue(adversarial_result['Model'])
            lower = JuMP.getobjectivebound(adversarial_result['Model'])

            if not np.isnan(upper):
                starting_point = adversarial

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
                #print('PyTorch Starting point label (1-indexed): ', utils.get_labels(self.predict, torch.unsqueeze(starting_point, dim=0))[0] + 1)
                starting_point = starting_point.detach().cpu().numpy()

            adversarial, adversarial_result = self.mip_attack(
                image, label, heuristic_starting_point=starting_point)
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
