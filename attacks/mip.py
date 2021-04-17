import logging
import time

import advertorch
import numpy as np
import torch
import torch.nn as nn

import torch_utils
import utils

logger = logging.getLogger(__name__)

# Required libraries:
# PyCall (installed from the pip package "julia")
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

# TODO: Fare un avviso che dice se ci sono differenze significative tra i vari parameters

class MIPAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    _pyjulia_installed = False

    def __init__(self, predict, p, targeted, tolerance=1e-6, clip_min=0,
                clip_max=1, main_parameters=None, tightening_parameters=None,
                exploration_main_parameters=None, exploration_tightening_parameters=None,
                correction_factor_schedule=None, main_attempts=1, retry_gap=1e-4,
                retry_absolute_gap=1e-5, original_if_failed=False):
        super().__init__(predict, None, clip_min, clip_max)

        if main_parameters is None:
            main_parameters = dict()
        if tightening_parameters is None:
            tightening_parameters = dict()
        if exploration_main_parameters is None:
            exploration_main_parameters = dict()
        if exploration_tightening_parameters is None:
            exploration_tightening_parameters = dict()
        if correction_factor_schedule is None:
            correction_factor_schedule = [1.05]

        if not np.isposinf(p):
            raise NotImplementedError('MIPAttack only supports p=1, 2 or inf.')

        # Lazy import
        if not MIPAttack._pyjulia_installed:
            import julia
            julia.install()
            MIPAttack._pyjulia_installed = True

        self.p = p
        self.mip_model = sequential_to_mip(predict)
        self.targeted = targeted
        self.correction_factor_schedule = correction_factor_schedule
        self.main_attempts = main_attempts
        self.retry_gap = retry_gap
        self.retry_absolute_gap = retry_absolute_gap
        self.original_if_failed = original_if_failed
        self._checked_model = False

        if tolerance == 0:
            logger.warning('MIP\'s tolerance is set to 0. Given the possible numerical errors,'
                        ' it is likely that MIP\'s adversarials will be considered unsuccessful by Torch\'s'
                        ' model.')

        self.tolerance = tolerance

        for parameters in [main_parameters, tightening_parameters, exploration_main_parameters, exploration_tightening_parameters]:
            if 'TimeLimit' in parameters and parameters['TimeLimit'] == 0:
                parameters['TimeLimit'] = np.inf

        mip_gap = main_parameters['MIPGap'] if 'MIPGap' in main_parameters else 1e-4
        if mip_gap != retry_gap:
            logger.warning('Main Solver MIPGap differs from retry_gap: this might '
                            'interfere with the retry system.')

        mip_gap_abs = main_parameters['MIPGapAbs'] if 'MIPGapAbs' in main_parameters else 1e-5
        if mip_gap_abs != retry_absolute_gap:
            logger.warning('Main Solver MIPGapAbs differs from retry_absolute_gap: this might '
                            'interfere with the retry system.')

        from julia import Gurobi
        # TODO: Capire bene come funziona Gurobi.Env()
        self.main_solver = Gurobi.GurobiSolver(Gurobi.Env(),
                                                     OutputFlag=0,
                                                     **main_parameters)
        self.tightening_solver = Gurobi.GurobiSolver(Gurobi.Env(),
                                                     OutputFlag=0,
                                                     **tightening_parameters)
        self.exploration_main_solver = Gurobi.GurobiSolver(Gurobi.Env(),
                                                     OutputFlag=0,
                                                     **exploration_main_parameters)
        self.exploration_tightening_solver = Gurobi.GurobiSolver(Gurobi.Env(),
                                                     OutputFlag=0,
                                                     **exploration_tightening_parameters)


    def _check_model(self, image, threshold=1e-3):
        model_device = next(self.predict.parameters()).device
        torch_output = self.predict(torch.unsqueeze(torch.tensor(image).to(model_device), 0)).detach().cpu().numpy()

        image = image.transpose([1, 2, 0])
        image = np.expand_dims(image, 0)
        julia_output = self.mip_model(image)

        return np.max(np.abs(torch_output - julia_output)) <= threshold


    def perform_attempt(self, image, label, main_solver, tightening_solver, perturbation_size):
        from julia import MIPVerify
        from julia import JuMP

        assert perturbation_size is None or perturbation_size > 0

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

        image = image.transpose([1, 2, 0])
        image = np.expand_dims(image, 0)

        adversarial_result = MIPVerify.find_adversarial_example(
            self.mip_model, image, target_label, main_solver,
            norm_order=self.p, tolerance=self.tolerance,
            invert_target_selection=not self.targeted,
            tightening_solver=tightening_solver, pp=perturbation)

        adversarial = np.array(JuMP.getvalue(
            adversarial_result['PerturbedInput']))

        # if extra_dimension:
        #    adversarial = np.expand_dims(adversarial, -1)

        # Reconvert to channel-first
        adversarial = adversarial.transpose([0, 3, 1, 2])

        # Remove the batch dimension
        adversarial = adversarial.squeeze(axis=0)

        elapsed_time = time.clock() - start_time

        lower = JuMP.getobjectivebound(adversarial_result['Model'])
        upper = JuMP.getobjectivevalue(adversarial_result['Model'])

        if np.isnan(lower):
            lower = None
        if np.isnan(upper):
            upper = None

        # Unsuccessful execution, replace with None
        if np.any(np.isnan(adversarial)):
            assert upper is None
            adversarial = None

        return adversarial, lower, upper, elapsed_time


    def find_perturbation_size(self, image, label, original_perturbation_size):
        from julia import JuMP

        for attempt, correction_factor in enumerate(self.correction_factor_schedule):
            logger.debug('Exploration attempt %s.', attempt)
            _, _, upper, _ = self.perform_attempt(image, label, self.exploration_main_solver, self.exploration_tightening_solver, original_perturbation_size * correction_factor)

            if upper is not None:
                return upper

        return None


    def mip_attack(self, image, label, starting_point=None):
        from julia import JuMP
        device, dtype = image.device, image.dtype

        image = image.detach().cpu().numpy()
        label = label.detach().cpu().item()

        # Check that the MIP model actually matches the
        # PyTorch one
        if not self._checked_model:
            assert self._check_model(image, threshold=1e-3)
            #self._checked_model = True # TODO: Temporaneamente disabilitato

        perturbation_size = None
        if starting_point is not None:
            original_perturbation_size = np.linalg.norm(
                    (image - starting_point).flatten(), ord=np.inf).item()
            perturbation_size = self.find_perturbation_size(image, label, original_perturbation_size)

        adversarial = None
        lower = None
        upper = None
        solve_time = None

        for attempt in range(self.main_attempts):
            logger.debug('Main attempt %s.', attempt)

            adversarial, lower, upper, solve_time = self.perform_attempt(image, label, self.main_solver, self.tightening_solver, perturbation_size=perturbation_size)

            if upper is not None:
                assert perturbation_size is None or upper <= perturbation_size
                assert upper > 0
                assert lower > 0
                assert upper >= lower

                absolute_gap = upper - lower
                gap = 0 if absolute_gap == 0 else (upper - lower) / upper
                # TODO: < o <= ?
                if absolute_gap < self.retry_absolute_gap or gap < self.retry_gap:
                    break

                perturbation_size = upper

        # Must have been run at least once
        assert solve_time is not None

        if adversarial is not None:
            adversarial = torch.from_numpy(adversarial).to(device=device, dtype=dtype)

        return adversarial, lower, upper, solve_time


    def perturb(self, x, y=None):
        if not self.original_if_failed:
            raise RuntimeError('perturb requires original_if_failed to be True.')
        x, y = self._verify_and_process_inputs(x, y)

        adversarials = []

        for image, label in zip(x, y):
            adversarial, _, _, _ = self.mip_attack(image, label)

            # In case of complete failure, return the original image
            if adversarial is None:
                adversarial = image

            adversarials.append(adversarial)

        return utils.maybe_stack(adversarials, x.shape[1:], device=x.device)


    # TODO: Rinominare
    def perturb_advanced(self, x, y=None, starting_points=None):
        x, y = self._verify_and_process_inputs(x, y)

        adversarials = []
        lower_bounds = []
        upper_bounds = []
        solve_times = []

        if starting_points is None:
            starting_points = [None] * len(x)

        for image, label, starting_point in zip(x, y, starting_points):
            if starting_point is not None:
                #print('PyTorch Starting point label (1-indexed): ', utils.get_labels(self.predict, torch.unsqueeze(starting_point, dim=0))[0] + 1)
                starting_point = starting_point.detach().cpu().numpy()

            adversarial, lower, upper, solve_time = self.mip_attack(
                image, label, starting_point=starting_point)

            from julia import JuMP

            if adversarial is None and self.original_if_failed:
                adversarial = image

            adversarials.append(adversarial)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            solve_times.append(solve_time)

        return adversarials, lower_bounds, upper_bounds, solve_times
