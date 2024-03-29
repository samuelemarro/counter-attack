import logging
import os
from pathlib import Path
import tempfile
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

SIMILARITY_THRESHOLD = 1e-5
OUTPUT_SIMILARITY_THRESHOLD = 1e-5
ATTACK_IMPROVEMENT_THRESHOLD = 1e-5

# See https://www.gurobi.com/documentation/9.1/refman/attributes.html
GUROBI_MODEL_ATTRIBUTES = [
    ('NumVars', 'int'),
    ('NumConstrs', 'int'),
    ('NumSOS', 'int'),
    ('NumQConstrs', 'int'),
    ('NumGenConstrs', 'int'),
    ('NumNZs', 'int'),
    ('DNumNZs', 'double'),
    ('NumQNZs', 'int'),
    ('NumQCNZs', 'int'),
    ('NumIntVars', 'int'),
    ('NumBinVars', 'int'),
    ('NumPWLObjVars', 'int'),
    ('ModelName', 'string'),
    ('ModelSense', 'int'),
    ('ObjCon', 'double'),
    # ('Fingerprint', 'int'),
    ('ObjVal', 'double'),
    ('ObjBound', 'double'),
    ('ObjBoundC', 'double'),
    ('PoolObjBound', 'double'),
    ('PoolObjVal', 'double'),
    ('MIPGap', 'double'),
    ('Runtime', 'double'),
    ('Status', 'int'),
    ('SolCount', 'int'),
    ('IterCount', 'double'),
    ('BarIterCount', 'int'),
    ('NodeCount', 'double'),
    ('IsMIP', 'int'),
    ('IsQP', 'int'),
    ('IsQCP', 'int'),
    ('IsMultiObj', 'int'),
    ('IISMinimal', 'int'),
    ('MaxCoeff', 'double'),
    ('MinCoeff', 'double'),
    ('MaxBound', 'double'),
    ('MinBound', 'double'),
    ('MaxObjCoeff', 'double'),
    ('MinObjCoeff', 'double'),
    ('MaxRHS', 'double'),
    ('MinRHS', 'double'),
    ('MaxQCCoeff', 'double'),
    ('MinQCCoeff', 'double'),
    ('MaxQCLCoeff', 'double'),
    ('MinQCLCoeff', 'double'),
    ('MaxQCRHS', 'double'),
    ('MinQCRHS', 'double'),
    ('MaxQObjCoeff', 'double'),
    ('MinQObjCoeff', 'double'),
    ('FarkasProof', 'double'),
    ('Kappa', 'double'),
    ('KappaExact', 'double')
]

GUROBI_QUALITY_ATTRIBUTES = [
    ('BoundVio', 'double'),
    ('BoundSVio', 'double'),
    ('BoundVioIndex', 'int'),
    ('BoundSVioIndex', 'int'),
    ('BoundVioSum', 'double'),
    ('BoundSVioSum', 'double'),
    ('ConstrVio', 'double'),
    ('ConstrSVio', 'double'),
    ('ConstrVioIndex', 'int'),
    ('ConstrSVioIndex', 'int'),
    ('ConstrVioSum', 'double'),
    ('ConstrSVioSum', 'double'),
    ('ConstrResidual', 'double'),
    ('ConstrSResidual', 'double'),
    ('ConstrResidualIndex', 'int'),
    ('ConstrSResidualIndex', 'int'),
    ('ConstrResidualSum', 'double'),
    ('ConstrSResidualSum', 'double'),
    ('DualVio', 'double'),
    ('DualSVio', 'double'),
    ('DualVioIndex', 'double'),
    ('DualSVioIndex', 'int'),
    ('DualVioSum', 'double'),
    ('DualSVioSum', 'double'),
    ('DualResidual', 'double'),
    ('DualSResidual', 'double'),
    ('DualResidualIndex', 'int'),
    ('DualSResidualIndex', 'int'),
    ('DualResidualSum', 'double'),
    ('DualSResidualSum', 'double'),
    ('ComplVio', 'double'),
    ('ComplVioIndex', 'int'),
    ('ComplVioSum', 'double'),
    ('IntVio', 'double'),
    ('IntVioIndex', 'int'),
    ('IntVioSum', 'double')
]

GUROBI_VARIABLE_ATTRIBUTES = [
    ('LB', 'double'),
    ('UB', 'double'),
    ('Obj', 'double'),
    ('VType', 'char'),
    ('VarName', 'string'),
    ('VTag', 'string'),
    ('X', 'double'),
    ('Xn', 'double'),
    ('RC', 'double'),
    ('BarX', 'double'),
    ('Start', 'double'),
    ('VarHintVal', 'double'),
    ('VarHintPri', 'int'),
    ('BranchPriority', 'int'),
    ('Partition', 'int'),
    ('VBasis', 'int'),
    ('PStart', 'double'),
    ('IISLB', 'int'),
    ('IISUB', 'int'),
    ('PWLObjCvx', 'int'),
    ('SAObjLow', 'double'),
    ('SAObjUp', 'double'),
    ('SALBLow', 'double'),
    ('SALBUp', 'double'),
    ('SAUBLow', 'double'),
    ('SAUBUp', 'double'),
    ('UnbdRay', 'double')
]

GUROBI_CONSTRAINT_ATTRIBUTES = [
    # ('Sense', 'char'),
    # ('RHS', 'double'),
    # ('ConstrName', 'string'),
    # ('CTag', 'string'),
    ('Pi', 'double'),
    ('Slack', 'double'),
    ('CBasis', 'int'),
    ('DStart', 'double'),
    ('Lazy', 'int'),
    ('IISConstr', 'int'),
    ('SARHSLow', 'double'),
    ('SARHSUp', 'double'),
    #('FarkasDual', 'double') # Dropped due to sometimes causing segfaults
]

def module_to_mip(module, flattened_input):
    from julia import MIPVerify

    # MIPVerify.Flatten removes the batch dimension, so we need
    # to handle the layers after MIPVerify.Flatten differently

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    if isinstance(module, nn.Linear):
        # PyTorch stores the weights W so that matmul(x, W.T) is possible,
        # which is equivalent to matmul(W, x). MIPVerify computes matmul(W.T, x).
        # We therefore transpose W so that MIPVerify will re-transpose it back to its
        # original form
        converted = MIPVerify.Linear(
            to_numpy(module.weight).transpose(),
            to_numpy(module.bias)
        )
    elif isinstance(module, nn.Flatten):
        assert not flattened_input
        assert module.end_dim == -1

        # PyTorch uses the BCHW format and reads row-first (backwards), so the order is WHCB
        # Julia uses the BHWC format and reads column-first (straightforward), so in order to read it
        # in the same way as PyTorch, we have to permute the dimensions to WHCB using the permutation:
        # [3, 2, 4, 1]
        converted = MIPVerify.Flatten(4, [3, 2, 4, 1])
    elif isinstance(module, nn.ReLU):
        # MIPVerify allows passing a custom tightening algorithm
        # to ReLU. By not passing it, ReLU will use the model's
        # tightening algorithm
        converted = MIPVerify.ReLU()
    elif isinstance(module, torch_utils.MaskedReLU):
        always_zero = to_numpy(module.always_zero.data)
        always_linear = to_numpy(module.always_linear.data)
        assert always_zero.shape == always_linear.shape
        assert not np.any(always_zero & always_linear)

        always_zero = always_zero.astype(np.float)
        always_linear = always_linear.astype(np.float)

        logger.debug('Adding %s always_zero elements and %s always_linear elements (out of %s elements).',
            np.sum(always_zero), np.sum(always_linear), np.prod(always_zero.shape))

        mask = np.zeros_like(always_zero)
        mask += always_linear
        mask -= always_zero

        assert np.all(np.equal(mask, 0) | np.equal(mask, 1) | np.equal(mask, -1))
        assert len(mask.shape) == 1 or len(mask.shape) == 3

        if len(mask.shape) == 3:
            assert not flattened_input

            # PyJulia automatically converts from Python's row-major order to Julia's
            # column-major one. We only need to convert from PyTorch's CHW to Julia's HWC
            mask = mask.transpose([1, 2, 0])

        if not flattened_input:
            # If the input is not flattened, the mask needs to have
            # a batch dimension
            mask = np.expand_dims(mask, 0)

        # Similarly to ReLU, MaskedReLU also supports passing a custom
        # tightening algorithm
        converted = MIPVerify.MaskedReLU(mask)
    elif isinstance(module, torch_utils.Normalisation):
        assert not flattened_input

        mean = np.squeeze(to_numpy(module.mean))
        std = np.squeeze(to_numpy(module.std))

        assert len(mean.shape) <= 1
        assert len(std.shape) <= 1

        # Expand zero-length mean and std
        if len(mean.shape) == 0:
            mean = [mean.item()]
        if len(std.shape) == 0:
            std = [std.item()]

        # No need for transpositions, we're only working with the
        # channel dimension
        converted = MIPVerify.Normalize(mean, std)
    elif isinstance(module, nn.Conv2d):
        # PyTorch follows the filter format:
        # (out_channels, in_channels / groups, kernel_size[0], kernel_size[1])
        # where kernel_size[0] = filter_height and kernel_size[1] = filter_width
        # MIP expects the following format:
        # (filter_height, filter_width, in_channels, out_channels)

        if not all(x == 1 for x in module.dilation):
            raise NotImplementedError(
                f'MIP only supports (1, 1)-style dilation. Received: {module.dilation}.')
        if not all(x == 0 for x in module.output_padding):
            raise NotImplementedError(
                f'MIP does not support output padding. Received: {module.output_padding}.')
        if module.groups != 1:
            raise NotImplementedError(
                f'MIP only supports convolutions with groups=1. Received: {module.groups}.')
        if len(set(module.stride)) > 1:
            raise NotImplementedError(
                f'MIP only supports striding along all dimensions with the same value. Received: {module.stride}.')

        if isinstance(module.stride, tuple):
            stride = module.stride[0]
        else:
            stride = module.stride

        filter_ = to_numpy(module.weight)

        # Transpose the filter to match MIPVerify
        filter_ = np.transpose(filter_, [2, 3, 1, 0])

        padding = module.padding
        bias = to_numpy(module.bias)

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

    converted_layers = []
    flattened_input = False

    for layer in layers:
        converted_layers.append(module_to_mip(layer, flattened_input))

        if isinstance(layer, nn.Flatten):
            flattened_input = True

    conversion_time = time.time()

    return MIPVerify.Sequential(converted_layers, f'Converted network ({conversion_time})')

class MIPAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    _pyjulia_installed = False

    def __init__(self, predict, p, targeted, tolerance=1e-6, clip_min=0,
                clip_max=1, main_parameters=None, tightening_parameters=None,
                exploration_main_parameters=None, exploration_tightening_parameters=None,
                correction_factor_schedule=None, main_attempts=1, retry_gap=1e-4,
                retry_absolute_gap=1e-5, original_if_failed=False, seed=None):
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

        if seed is not None:
            # Global seed provided: set it for all solvers that do not
            # have a custom seed
            parameter_pairs = [
                (main_parameters, 'main solver'),
                (tightening_parameters, 'tightening solver'),
                (exploration_main_parameters, 'exploration main solver'),
                (exploration_tightening_parameters, 'exploration tightening solver')
            ]

            for parameter_dict, name in parameter_pairs:
                if 'Seed' in parameter_dict:
                    logger.info('Found custom seed for the %s.', name)
                else:
                    logger.info('No custom seed provided for the %s. Using '
                                'the global one.', name)
                    parameter_dict['Seed'] = seed

        if not np.isposinf(p):
            raise NotImplementedError('MIPAttack only supports p=inf.')

        # Lazy import
        if not MIPAttack._pyjulia_installed:
            import julia
            julia.install()
            MIPAttack._pyjulia_installed = True

        if tolerance == 0:
            logger.warning('MIP\'s tolerance is set to 0. Given the possible numerical errors, '
                        'it is likely that MIP\'s adversarials will be considered unsuccessful by '
                        'Torch\'s model.')

        for parameters in [main_parameters, tightening_parameters, exploration_main_parameters, exploration_tightening_parameters]:
            # TimeLimit=0 means that it does not have a time limit
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

        self.p = p
        self.targeted = targeted
        self.tolerance = tolerance
        self.correction_factor_schedule = correction_factor_schedule
        self.main_attempts = main_attempts
        self.retry_gap = retry_gap
        self.retry_absolute_gap = retry_absolute_gap
        self.original_if_failed = original_if_failed
        self.mip_model = sequential_to_mip(predict)

        self.main_parameters = main_parameters
        self.tightening_parameters = tightening_parameters
        self.exploration_main_parameters = exploration_main_parameters
        self.exploration_tightening_parameters = exploration_tightening_parameters

    def _get_solver(self, original_parameters, attempt, log_file):
        from julia import Gurobi
        parameters = dict(original_parameters)

        if 'Seed' in parameters:
            parameters['Seed'] = parameters['Seed'] + attempt

        return Gurobi.GurobiSolver(OutputFlag=1, InfUnbdInfo=1, DisplayInterval=1, LogFile=str(log_file), **parameters)

    def _check_model(self, image, threshold=OUTPUT_SIMILARITY_THRESHOLD):
        model_device = next(self.predict.parameters()).device
        torch_image = torch.unsqueeze(torch.tensor(image).to(model_device), 0)

        with torch.no_grad():
            torch_output = self.predict(torch_image).cpu()
            assert torch_output.grad is None
            torch_output = torch_output.numpy()

        image = image.transpose([1, 2, 0])
        image = np.expand_dims(image, 0)
        mip_output = self.mip_model(image)

        return np.max(np.abs(torch_output - mip_output)) <= threshold


    def _run_mipverify(self, image, label, main_parameters, tightening_parameters, attempt, perturbation_size, gurobi_log_dir):
        run_mipverify_start_timestamp = time.time()

        assert not gurobi_log_dir.exists()
        gurobi_log_dir.mkdir(parents=True)

        main_log_file = str(gurobi_log_dir / 'main.log')
        main_solver = self._get_solver(main_parameters, attempt, main_log_file)

        tightening_log_file = str(gurobi_log_dir / 'tightening.log')
        tightening_solver = self._get_solver(tightening_parameters, attempt, tightening_log_file)

        julia_import_start_timestamp = time.time()

        from julia import MIPVerify
        from julia import JuMP
        from julia import Gurobi
        from julia import Main
        Main.include('mip_interface.jl')

        julia_import_end_timestamp = time.time()

        assert perturbation_size is None or perturbation_size > 0

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

        # Convert to HWC
        mip_image = image.transpose([1, 2, 0])

        # MIP expects a batch dimension
        mip_image = np.expand_dims(mip_image, 0)

        adversarial_result = Main.find_adversarial_example(
            self.mip_model, mip_image, target_label, main_solver,
            norm_order=self.p, tolerance=self.tolerance,
            invert_target_selection=not self.targeted,
            tightening_solver=tightening_solver, pp=perturbation,
            rebuild=True, cache_model=False)

        adversarial = np.array(JuMP.getvalue(
            adversarial_result['PerturbedInput']))

        assert mip_image.shape == adversarial.shape

        # Reconvert to BCHW
        adversarial = adversarial.transpose([0, 3, 1, 2])

        # Remove the batch dimension
        adversarial = adversarial.squeeze(axis=0)

        assert image.shape == adversarial.shape


        lower = JuMP.getobjectivebound(adversarial_result['Model'])
        upper = JuMP.getobjectivevalue(adversarial_result['Model'])

        if np.isnan(lower):
            lower = None
        else:
            assert lower >= 0

        if np.isnan(upper):
            upper = None
        else:
            assert upper > 0

        if upper is not None and lower is not None:
            assert upper >= lower

        if np.any(np.isnan(adversarial)):
            # Unsuccessful execution, replace with None
            assert upper is None
            adversarial = None
        else:
            # Successful execution, check that upper is (within numerical precision)
            # equal to linf_distance
            linf_distance = np.max(np.abs(image - adversarial))
            assert np.abs(linf_distance - upper) < SIMILARITY_THRESHOLD

        extra_info = {
            'times' : {
                'julia' : {
                    'gurobi_solve_time' : adversarial_result['GurobiSolveTime'],
                    'find_adversarial_example' : {
                        'start_timestamp' : adversarial_result['FindAdversarialExampleStartTimestamp'],
                        'end_timestamp' : adversarial_result['FindAdversarialExampleEndTimestamp']
                    },
                    'model_building' : {
                        'start_timestamp' : adversarial_result['ModelBuildingStartTimestamp'],
                        'end_timestamp' : adversarial_result['ModelBuildingEndTimestamp']
                    },
                    'wall_clock' : {
                        'start_timestamp' : adversarial_result['WallClockStartTimestamp'],
                        'end_timestamp' : adversarial_result['WallClockEndTimestamp']
                    }
                }
            }
        }

        inner_model = JuMP.internalmodel(adversarial_result['Model']).inner

        def get_gurobi_attribute(name, attribute_type):
            if attribute_type == 'int':
                return Gurobi.get_intattr(inner_model, name)
            elif attribute_type == 'double':
                return Gurobi.get_dblattr(inner_model, name)
            elif attribute_type == 'string':
                return Gurobi.get_strattr(inner_model, name)
            else:
                raise NotImplementedError

        def get_gurobi_array_attribute(name, attribute_type, length):
            start = 1 # 1-indexed
            if attribute_type == 'int':
                # The default implementation doesn't work, so we use
                # a workaround
                return Main.get_intattrarray_workaround(inner_model, name, start, length)
            elif attribute_type == 'double':
                return Gurobi.get_dblattrarray(inner_model, name, start, length)
            elif attribute_type == 'char':
                characters = Gurobi.get_charattrarray(inner_model, name, start, length)
                return [Main.string(c) for c in characters]
            elif attribute_type == 'string':
                return Gurobi.get_strattrarray(inner_model, name, start, length)
            else:
                raise NotImplementedError

        attribute_retrieval_start_timestamp = time.time()

        extra_info['gurobi_attributes'] = {
                'model' : {},
                'quality' : {},
                'variable' : {},
                'constraint' : {}
        }

        for name, attribute_type in GUROBI_MODEL_ATTRIBUTES:
            try:
                extra_info['gurobi_attributes']['model'][name] = get_gurobi_attribute(name, attribute_type)
            except RuntimeError as e:
                logger.info('Skipped attribute %s, reason: %s', name, e)
                extra_info['gurobi_attributes']['model'][name] = None

        for name, attribute_type in GUROBI_QUALITY_ATTRIBUTES:
            try:
                extra_info['gurobi_attributes']['quality'][name] = get_gurobi_attribute(name, attribute_type)
            except RuntimeError as e:
                logger.info('Skipped attribute %s, reason: %s', name, e)
                extra_info['gurobi_attributes']['quality'][name] = None

        var_count = extra_info['gurobi_attributes']['model']['NumVars']
        for name, attribute_type in GUROBI_VARIABLE_ATTRIBUTES:
            try:
                extra_info['gurobi_attributes']['variable'][name] = get_gurobi_array_attribute(name, attribute_type, var_count)
            except Exception as e:
                logger.info('Skipped attribute %s, reason: %s', name, e)
                extra_info['gurobi_attributes']['variable'][name] = None
        
        constraint_count = extra_info['gurobi_attributes']['model']['NumConstrs']
        for name, attribute_type in GUROBI_CONSTRAINT_ATTRIBUTES:
            try:
                extra_info['gurobi_attributes']['constraint'][name] = get_gurobi_array_attribute(name, attribute_type, constraint_count)
            except Exception as e:
                logger.info('Skipped attribute %s, reason: %s', name, e)
                extra_info['gurobi_attributes']['constraint'][name] = None

        extra_info['logs'] = {}

        with open(main_log_file, 'r') as main_f:
            extra_info['logs']['main'] = main_f.read()
        with open(tightening_log_file, 'r') as tightening_f:
            extra_info['logs']['tightening'] = tightening_f.read()

        run_mipverify_end_timestamp = attribute_retrieval_end_timestamp = time.time()

        extra_info['times']['python'] = {
            'attribute_retrieval' : {
                'start_timestamp' : attribute_retrieval_start_timestamp,
                'end_timestamp' : attribute_retrieval_end_timestamp
            },
            'run_mipverify' : {
                'start_timestamp' : run_mipverify_start_timestamp,
                'end_timestamp' : run_mipverify_end_timestamp
            },
            'julia_import' : {
                'start_timestamp' : julia_import_start_timestamp,
                'end_timestamp' : julia_import_end_timestamp
            }
        }

        return adversarial, lower, upper, extra_info


    def _mip_success(self, lower, upper):
        assert upper >= lower

        absolute_gap = upper - lower

        # absolute_gap = 0 ==> gap = 0
        gap = 0 if absolute_gap == 0 else (upper - lower) / upper

        # Following Gurobi's convention, we use < instead of <=
        return absolute_gap < self.retry_absolute_gap or gap < self.retry_gap


    def _find_perturbation_size(self, image, label, original_perturbation_size, gurobi_log_dir):
        from julia import JuMP

        exploration_extra_infos = []

        for attempt, correction_factor in enumerate(self.correction_factor_schedule):
            logger.info('Exploration attempt %s (correction factor: %s).', attempt, correction_factor)
            adversarial, lower, upper, extra_info = self._run_mipverify(
                image,
                label,
                self.exploration_main_parameters,
                self.exploration_tightening_parameters,
                attempt,
                original_perturbation_size * correction_factor,
                gurobi_log_dir / 'exploration_run' / f'attempt_{attempt}')
            
            exploration_extra_infos.append(extra_info)

            if upper is not None:
                # Found an upper bound, which was our objective
                logger.info('Exploration was successful (correction factor: %s).', correction_factor)

                if self._mip_success(lower, upper):
                    # Found a successful result: return it
                    return upper, exploration_extra_infos, (adversarial, lower, upper)
                else:
                    # Found an upper bound without a successful result:
                    # return only the upper bound
                    return upper, exploration_extra_infos, None

        logger.info('Exploration failed.')
        # Failed to find an upper bound: return None
        return None, exploration_extra_infos, None


    def _mip_attack(self, image, label, gurobi_log_dir, starting_point=None):
        from julia import JuMP

        extra_info = {
            'exploration' : None,
            'main' : None
        }

        start_time = time.time()

        assert image.device == label.device

        # Store the device and dtype for restoring
        device, dtype = image.device, image.dtype

        # Convert to NumPy

        image = image.detach().cpu().numpy()
        label = label.detach().cpu().item()

        if starting_point is not None:
            starting_point = starting_point.detach().cpu().numpy()

        # Check that the MIP model matches the PyTorch
        # one (for this image)
        assert self._check_model(image, threshold=OUTPUT_SIMILARITY_THRESHOLD)

        perturbation_size = None

        if starting_point is not None:
            # Use the starting point to compute an
            # initial perturbation size
            original_perturbation_size = np.linalg.norm(
                    (image - starting_point).flatten(), ord=np.inf).item()

            # Find a perturbation size
            perturbation_size, exploration_extra_infos, successful_result = self._find_perturbation_size(image, label, original_perturbation_size, gurobi_log_dir)

            extra_info['exploration'] = exploration_extra_infos

            if successful_result is not None:
                # The exploration was sufficient to find a successful
                # bound, skip the entire main loop

                adversarial, lower, upper = successful_result

                assert adversarial is not None

                # Convert to the stored device and dtype
                adversarial = torch.from_numpy(adversarial).to(device=device, dtype=dtype)

                elapsed_time = time.time() - start_time
                return adversarial, lower, upper, elapsed_time, extra_info

        adversarial = None
        lower = None
        upper = None
        extra_info['main'] = []

        for attempt in range(self.main_attempts):
            logger.info('Main attempt %s.', attempt)

            adversarial, lower, upper, main_extra_info = self._run_mipverify(
                image,
                label,
                self.main_parameters,
                self.tightening_parameters,
                attempt,
                perturbation_size,
                gurobi_log_dir / 'main_run' / f'attempt_{attempt}')

            extra_info['main'].append(main_extra_info)

            if upper is not None:
                # Found an upper bound
                assert perturbation_size is None or upper <= perturbation_size + SIMILARITY_THRESHOLD
                assert adversarial is not None
                assert upper > 0
                assert lower >= 0
                assert upper >= lower

                linf_distance = np.max(np.abs(image - adversarial))
                assert np.abs(linf_distance - upper) < SIMILARITY_THRESHOLD

                if self._mip_success(lower, upper):
                    # The bounds were successful, avoid further attempts
                    assert perturbation_size is None or linf_distance <= perturbation_size + ATTACK_IMPROVEMENT_THRESHOLD
                    break

                # Failure, use the new upper as the perturbation size
                perturbation_size = upper

        # Must have been run at least once
        assert main_extra_info is not None

        # Convert to the stored device and dtype
        if adversarial is not None:
            adversarial = torch.from_numpy(adversarial).to(device=device, dtype=dtype)

        elapsed_time = time.time() - start_time
        return adversarial, lower, upper, elapsed_time, extra_info


    def perturb(self, x, y=None):
        if not self.original_if_failed:
            raise RuntimeError('perturb requires original_if_failed to be True.')

        adversarials, _, _, _, _ = self.perturb_advanced(x, y=y)

        return utils.maybe_stack(adversarials, x.shape[1:], device=x.device)


    def perturb_advanced(self, x, y=None, starting_points=None, gurobi_log_dir=None):
        attack_batch_start_timestamp = time.time()

        if gurobi_log_dir is None:
            gurobi_log_dir = Path(tempfile.mkdtemp())
        else:
            gurobi_log_dir = Path(gurobi_log_dir)
            if gurobi_log_dir.exists():
                raise ValueError('gurobi_log_dir must not exist.')

        x, y = self._verify_and_process_inputs(x, y)

        assert len(x) == len(y)

        adversarials = []
        lower_bounds = []
        upper_bounds = []
        elapsed_times = []
        extra_infos = []

        if starting_points is None:
            starting_points = [None] * len(x)

        assert len(x) == len(starting_points)

        for index, (image, label, starting_point) in enumerate(zip(x, y, starting_points)):
            attack_element_start_timestamp = time.time()
            adversarial, lower, upper, elapsed_time, extra_info = self._mip_attack(
                image, label, gurobi_log_dir / f'item_{index}', starting_point=starting_point)

            if adversarial is None:
                if self.original_if_failed:
                    # Return the original image
                    adversarial = image
            else:
                assert lower is not None
                assert upper is not None
                assert image.shape == adversarial.shape

                # Check that the adversarial distance is equal to
                # the upper bound (within numerical precision)
                linf_distance = torch.max(torch.abs(image - adversarial)).cpu().numpy()
                assert np.abs(linf_distance - upper) < SIMILARITY_THRESHOLD

                if self._mip_success(lower, upper) and starting_point is not None:
                    # Check that the found adversarial is at most as close as the starting point
                    # (within numerical precision)
                    starting_point_linf_distance = torch.max(torch.abs(image - starting_point)).cpu().numpy()
                    assert linf_distance <= starting_point_linf_distance + ATTACK_IMPROVEMENT_THRESHOLD

            attack_element_end_timestamp = time.time()
            extra_info['overall'] = {
                'times' : {
                    'attack_batch' : {
                        'start_timestamp' : attack_batch_start_timestamp
                    },
                    'attack_element' : {
                        'start_timestamp' : attack_element_start_timestamp,
                        'end_timestamp' : attack_element_end_timestamp
                    }
                }
            }

            adversarials.append(adversarial)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            elapsed_times.append(elapsed_time)
            extra_infos.append(extra_info)

        attack_batch_end_timestamp = time.time()

        for extra_info in extra_infos:
            extra_info['overall']['times']['attack_batch']['end_timestamp'] = attack_batch_end_timestamp

        return adversarials, lower_bounds, upper_bounds, elapsed_times, extra_infos
