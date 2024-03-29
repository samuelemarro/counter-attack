import logging
import warnings

import advertorch
from advertorch.utils import clamp, replicate_input, tanh_rescale, torch_arctanh, to_one_hot
import torch
import torch.optim as optim

import attacks
import utils

logger = logging.getLogger(__name__)

ONE_MINUS_EPS = 0.999999
TARGET_MULT = 10000.0
MAX_DISTANCE = 100000.0
SMALL_LOSS_COEFFICIENT = 0.0001

def get_carlini_linf_attack(target_model, num_classes, use_best_sample, cuda_optimized=True, **kwargs):
    if cuda_optimized:
        return CarliniWagnerCUDALinfAttack(target_model, num_classes, use_best_sample=use_best_sample, **kwargs)
    else:
        # Remove CUDA-specific arguments
        kwargs.pop('tau_check', None)
        kwargs.pop('const_check', None)
        kwargs.pop('inner_check', None)

        return CarliniWagnerCPULinfAttack(target_model, num_classes, use_best_sample=use_best_sample, **kwargs)

class CarliniWagnerLinfAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    """
    The Carlini and Wagner LInfinity Attack, https://arxiv.org/abs/1608.04644
    This is a base class for the CPU and CUDA versions.

    :param predict: forward pass function (pre-softmax).
    :param num_classes: number of classes.
    :param min_tau: the minimum value of tau.
    :param initial_tau: the initial value of tau.
    :param tau_factor: the decay rate of tau (between 0 and 1)
    :param initial_const: initial value of the constant c.
    :param max_const: the maximum value of the constant c.
    :param const_factor: the rate of growth of the constant c.
    :param reduce_const: if True, the inital value of c is halved every
        time tau is reduced.
    :param warm_start: if True, use the previous adversarials as starting point
        for the next iteration.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm.
    :param max_iterations: the maximum number of iterations.
    :param abort_early: if set to true, abort early if getting stuck in local
        min.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function.
    :param use_best_sample: pass best_sample tracking data to the model.
    """
    def __init__(self, predict, num_classes, min_tau=1 / 256,
                 initial_tau=1, tau_factor=0.9, initial_const=1e-5,
                 max_const=20, const_factor=2, reduce_const=False,
                 warm_start=True, targeted=False, learning_rate=5e-3,
                 max_iterations=1000, abort_early=True, clip_min=0.,
                 clip_max=1., loss_fn=None, use_best_sample=False):
        """Carlini Wagner LInfinity Attack implementation in pytorch."""
        if loss_fn is not None:
            warnings.warn(
                "This Attack currently does not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        if isinstance(predict, attacks.BestSampleWrapper) and not use_best_sample:
            logger.warning('Using best sampling without use_best_sample=True. '
                           'This might lead to incorrect results.')

        loss_fn = None
        super().__init__(predict, loss_fn, clip_min=clip_min, clip_max=clip_max)

        self.num_classes = num_classes
        self.min_tau = min_tau
        self.initial_tau = initial_tau
        self.tau_factor = tau_factor
        self.initial_const = initial_const
        self.max_const = max_const
        self.const_factor = const_factor
        self.reduce_const = reduce_const
        self.warm_start = warm_start
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.use_best_sample = use_best_sample

    def _get_arctanh_x(self, x):
        # Carlini's original implementation uses a slightly different formula because
        # the image space is [-0.5, 0.5] instead of [clip_min, clip_max]
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=0., max=1.) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def _outputs(self, x, active_mask=None, filter_=None):
        if self.use_best_sample:
            return self.predict(x, active_mask=active_mask, filter_=filter_)
        else:
            return self.predict(x)

    def _outputs_and_loss(self, x, modifiers, starting_atanh, y, const, taus, active_mask=None, filter_=None):
        # If you're comparing with Carlini's original implementation, x
        # is the name that has been given to tf.tanh(timg)/2, while
        # adversarials is the name that has been given to tf.tanh(modifier + simg)/2, aka newimg
        adversarials = tanh_rescale(
            starting_atanh + modifiers, self.clip_min, self.clip_max)

        assert x.shape == adversarials.shape

        outputs = self._outputs(adversarials, active_mask=active_mask, filter_=filter_)
        assert outputs.shape == (adversarials.shape[0], self.num_classes)

        y_onehot = to_one_hot(y, self.num_classes).float()
        assert y_onehot.shape == outputs.shape

        real = (y_onehot * outputs).sum(dim=1)

        other = ((1.0 - y_onehot) * outputs - (y_onehot * TARGET_MULT)
                 ).max(dim=1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.targeted:
            loss1 = torch.clamp(other - real, min=0.)
        else:
            loss1 = torch.clamp(real - other, min=0.)

        image_dimensions = tuple(range(1, len(x.shape)))

        # Reshape taus to [batch_size, 1, 1, 1] for broadcasting
        taus_shape = (len(x),) + (1,) * (len(x.shape) - 1)

        penalties = torch.clamp(
            torch.abs(adversarials - x) - taus.view(taus_shape), min=0)
        assert penalties.shape == x.shape

        loss2 = torch.sum(penalties, dim=image_dimensions)
        assert loss2.shape == loss1.shape

        losses = const * loss1 + loss2
        assert losses.shape == (len(x),)

        # losses is returned as a (batch_size,) vector to support abort_early
        # Only later it is converted to a scalar
        return outputs.detach(), losses

    def _successful(self, outputs, y):
        adversarial_labels = torch.argmax(outputs, dim=1)

        if self.targeted:
            return torch.eq(adversarial_labels, y)
        else:
            return ~torch.eq(adversarial_labels, y)

    def perturb(self, **kwargs):
        raise NotImplementedError

class CarliniWagnerCPULinfAttack(CarliniWagnerLinfAttack):
    """
    The Carlini and Wagner LInfinity Attack, https://arxiv.org/abs/1608.04644
    This version has been optimized for CPU execution.

    :param predict: forward pass function (pre-softmax).
    :param num_classes: number of classes.
    :param min_tau: the minimum value of tau.
    :param initial_tau: the initial value of tau.
    :param tau_factor: the decay rate of tau (between 0 and 1)
    :param initial_const: initial value of the constant c.
    :param max_const: the maximum value of the constant c.
    :param const_factor: the rate of growth of the constant c.
    :param reduce_const: if True, the inital value of c is halved every
        time tau is reduced.
    :param warm_start: if True, use the previous adversarials as starting point
        for the next iteration.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm.
    :param max_iterations: the maximum number of iterations.
    :param abort_early: if set to true, abort early if getting stuck in local
        min.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function.
    :param use_best_sample: pass best_sample tracking data to the model.
    """

    def __init__(self, predict, num_classes, min_tau=1 / 256,
                 initial_tau=1, tau_factor=0.9, initial_const=1e-5,
                 max_const=20, const_factor=2, reduce_const=False,
                 warm_start=True, targeted=False, learning_rate=5e-3,
                 max_iterations=1000, abort_early=True, clip_min=0.,
                 clip_max=1., loss_fn=None, use_best_sample=False):
        super().__init__(predict, num_classes, min_tau=min_tau,
                        initial_tau=initial_tau, tau_factor=tau_factor, initial_const=initial_const,
                        max_const=max_const, const_factor=const_factor, reduce_const=reduce_const,
                        warm_start=warm_start, targeted=targeted, learning_rate=learning_rate,
                        max_iterations=max_iterations, abort_early=abort_early, clip_min=clip_min,
                        clip_max=clip_max, loss_fn=loss_fn, use_best_sample=use_best_sample)

    def _run_attack(self, x, y, initial_const, taus, prev_adversarials, outer_active_mask):
        assert len(x) == len(taus)
        batch_size = len(x)
        computed_adversarials = x.clone().detach()

        if self.warm_start:
            starting_atanh = self._get_arctanh_x(prev_adversarials.clone())
        else:
            starting_atanh = self._get_arctanh_x(x.clone())

        modifiers = torch.nn.Parameter(torch.zeros_like(starting_atanh))

        # An array of booleans that stores which samples have not converged
        # yet
        active = torch.ones((batch_size,), dtype=torch.bool, device=x.device)
        optimizer = optim.Adam([modifiers], lr=self.learning_rate)

        const = initial_const
        
        # Used for best_sample tracking
        active_mask = outer_active_mask.clone()

        while torch.any(active) and const < self.max_const:
            # We add an extra iteration because adversarials
            # are not saved until the next iteration
            for _ in range(self.max_iterations + 1):
                # Only the elements of active_mask where outer_active_mask is True are changed
                active_mask[outer_active_mask] = active

                outputs, losses = self._outputs_and_loss(
                    x[active],
                    modifiers[active],
                    starting_atanh[active],
                    y[active],
                    const,
                    taus[active],
                    active_mask=active_mask)

                adversarials = tanh_rescale(
                    starting_atanh[active] + modifiers[active],
                    self.clip_min,
                    self.clip_max).detach()

                computed_adversarials[active] = adversarials

                # Update the modifiers
                # Note: this will update the modifiers of adversarials that might be
                # possibly dropped. This is not an issue, since adversarials are detached from
                # the gradient graph and saved before updating. In other words, the modifiers
                # will be updated, while the adversarials won't be (at least until the next iteration)
                total_loss = torch.sum(losses)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # If early aborting is enabled, drop successful
                # samples with a small loss (the current adversarials
                # are saved regardless of whether they are dropped)
                if self.abort_early:
                    successful = self._successful(outputs, y[active]).detach()
                    small_loss = losses < SMALL_LOSS_COEFFICIENT * const

                    active[active] = ~(successful & small_loss)

                    if not active.any():
                        break

            # Give more weight to the output loss
            const *= self.const_factor

        return computed_adversarials

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        x = replicate_input(x)
        batch_size = len(x)
        final_adversarials = x.clone()

        # An array of booleans that stores which samples have not converged
        # yet
        active = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

        initial_const = self.initial_const
        taus = torch.ones((batch_size,), device=x.device) * self.initial_tau

        # The previous adversarials. This is used to perform a "warm start"
        # during optimisation
        prev_adversarials = x.clone()

        while torch.any(active):
            new_adversarials = self._run_attack(
                x[active],
                y[active],
                initial_const,
                taus[active],
                prev_adversarials[active].clone(),
                outer_active_mask=active).detach()

            # Store the adversarials for the next iteration,
            # even if they failed
            prev_adversarials[active] = new_adversarials

            adversarial_outputs = self._outputs(new_adversarials, active_mask=active)
            successful = self._successful(adversarial_outputs, y[active]).detach()

            # If the Linf distance is lower than tau and the adversarial
            # is successful, use it as the new tau
            linf_distances = torch.max(
                torch.abs(new_adversarials - x[active]).flatten(1),
                dim=1)[0]
            assert linf_distances.shape == (len(new_adversarials),)

            linf_lower = linf_distances < taus[active]

            utils.replace_active(linf_distances,
                                 taus,
                                 active,
                                 linf_lower & successful)

            # Save the remaining adversarials
            utils.replace_active(new_adversarials,
                                 final_adversarials,
                                 active,
                                 successful)

            taus *= self.tau_factor

            if self.reduce_const:
                initial_const /= 2

            # Drop failed samples or with a low tau
            low_tau = taus[active] <= self.min_tau
            drop = low_tau | (~successful)
            active[active] = ~drop

        return final_adversarials


class CarliniWagnerCUDALinfAttack(CarliniWagnerLinfAttack):
    """
    The Carlini and Wagner LInfinity Attack, https://arxiv.org/abs/1608.04644.
    This version has been optimized for CUDA execution.

    :param predict: forward pass function (pre-softmax).
    :param num_classes: number of classes.
    :param min_tau: the minimum value of tau.
    :param initial_tau: the initial value of tau.
    :param tau_factor: the decay rate of tau (between 0 and 1)
    :param initial_const: initial value of the constant c.
    :param max_const: the maximum value of the constant c.
    :param const_factor: the rate of growth of the constant c.
    :param reduce_const: if True, the inital value of c is halved every
        time tau is reduced.
    :param warm_start: if True, use the previous adversarials as starting point
        for the next iteration.
    :param targeted: if the attack is targeted.
    :param learning_rate: the learning rate for the attack algorithm.
    :param max_iterations: the maximum number of iterations.
    :param abort_early: if set to true, abort early if getting stuck in local
        min.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param loss_fn: loss function.
    :param use_best_sample: pass best_sample tracking data to the model.
    :param tau_check: how often the attack checks if it can stop
        inside the tau loop. The check will be performed every tau_check
        iterations. 0 disables checking.
    :param const_check: how often the attack checks if it can stop
        inside the const loop. The check will be performed every const_check
        iterations. 0 disables checking. Ignored if abort_early is False.
    :param inner_check: how often the attack checks if it can stop
        inside the inner loop. The check will be performed every
        inner_check iterations. 0 disables checking. Ignored if abort_early is False.
    :param update_inactive: if True, the samples will be updated even if they have
        been declared inactive.
    """

    def __init__(self, predict, num_classes, min_tau=1 / 256,
                 initial_tau=1, tau_factor=0.9, initial_const=1e-5,
                 max_const=20, const_factor=2, reduce_const=False,
                 warm_start=True, targeted=False, learning_rate=5e-3,
                 max_iterations=1000, abort_early=True, clip_min=0.,
                 clip_max=1., loss_fn=None, use_best_sample=False,
                 tau_check=1, const_check=1, inner_check=1000,
                 update_inactive=False):
        super().__init__(predict, num_classes, min_tau=min_tau,
                        initial_tau=initial_tau, tau_factor=tau_factor, initial_const=initial_const,
                        max_const=max_const, const_factor=const_factor, reduce_const=reduce_const,
                        warm_start=warm_start, targeted=targeted, learning_rate=learning_rate,
                        max_iterations=max_iterations, abort_early=abort_early, clip_min=clip_min,
                        clip_max=clip_max, loss_fn=loss_fn, use_best_sample=use_best_sample)

        self.tau_check = tau_check
        self.const_check = const_check
        self.inner_check = inner_check
        self.update_inactive = update_inactive

    def _run_attack(self, x, y, initial_const, taus, prev_adversarials, active):
        assert len(x) == len(taus)
        batch_size = len(x)
        computed_adversarials = x.clone().detach()

        if self.warm_start:
            starting_atanh = self._get_arctanh_x(prev_adversarials.clone())
        else:
            starting_atanh = self._get_arctanh_x(x.clone())

        modifiers = torch.nn.Parameter(torch.zeros_like(starting_atanh))

        # An array of booleans that stores which samples have not converged
        # yet
        optimizer = optim.Adam([modifiers], lr=self.learning_rate)

        const = initial_const

        j = 0
        stop_search = False

        while (not stop_search) and const < self.max_const:
            # We add an extra iteration because adversarials are
            # not saved until the next iteration
            for k in range(self.max_iterations + 1):
                # Note: unlike the CPU version, the CUDA version updates and calls the model
                # on all samples, including inactive ones. However, the filter_ parameter is designed
                # to force best_sample to only update active samples. This is counter-productive, but
                # it's necessary in order to have consistent CPU and CUDA implementations
                outputs, losses = self._outputs_and_loss(
                    x,
                    modifiers,
                    starting_atanh,
                    y,
                    const,
                    taus,
                    filter_=active)

                adversarials = tanh_rescale(
                    starting_atanh + modifiers,
                    self.clip_min,
                    self.clip_max).detach()

                replace = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

                if not self.update_inactive:
                    replace = replace & active

                computed_adversarials = utils.fast_boolean_choice(computed_adversarials, adversarials, replace)

                # Update the modifiers
                total_loss = torch.sum(losses)
                #total_loss = torch.sum(losses[active]) # Temp
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # If early aborting is enabled, drop successful
                # samples with a small loss (the current adversarials
                # are saved regardless of whether they are dropped)
                if self.abort_early:
                    successful = self._successful(outputs, y).detach()
                    small_loss = losses < SMALL_LOSS_COEFFICIENT * const

                    active = active & ~(successful & small_loss)

                    if self.inner_check != 0 and (k + 1) % self.inner_check == 0:
                        # Causes an implicit sync point
                        if not active.any():
                            # Break from both loops
                            stop_search = True
                            break

            if stop_search:
                break

            if self.abort_early and self.const_check != 0 and (j + 1) % self.const_check == 0:
                # Causes an implicit sync point
                if not active.any():
                    break

            # Give more weight to the output loss
            const *= self.const_factor

        return computed_adversarials

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        x = replicate_input(x)
        batch_size = len(x)
        final_adversarials = x.clone()

        # An array of booleans that stores which samples have not converged
        # yet
        active = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

        initial_const = self.initial_const
        taus = torch.ones((batch_size,), device=x.device) * self.initial_tau

        # The previous adversarials. This is used to perform a "warm start"
        # during optimisation
        prev_adversarials = x.clone()

        max_tau = self.initial_tau

        i = 0

        while max_tau > self.min_tau:
            new_adversarials = self._run_attack(
                x,
                y,
                initial_const,
                taus,
                prev_adversarials.clone(),
                active).detach()

            # Store the adversarials for the next iteration,
            # even if they failed
            prev_adversarials = new_adversarials

            adversarial_outputs = self._outputs(new_adversarials, filter_=active)
            successful = self._successful(adversarial_outputs, y).detach()

            # If the Linf distance is lower than tau and the adversarial
            # is successful, use it as the new tau
            linf_distances = torch.max(
                torch.abs(new_adversarials - x).flatten(1),
                dim=1)[0]
            linf_lower = linf_distances < taus

            taus = utils.fast_boolean_choice(taus, linf_distances, linf_lower & successful)

            # Save the remaining adversarials
            replace = successful

            if not self.update_inactive:
                replace = replace & active

            final_adversarials = utils.fast_boolean_choice(final_adversarials, new_adversarials, replace)

            taus *= self.tau_factor
            max_tau = taus.max()

            if self.reduce_const:
                initial_const /= 2

            # Drop failed samples or with a low tau
            low_tau = taus <= self.min_tau
            drop = low_tau | (~successful)
            active = active & (~drop)

            if self.tau_check != 0 and (i + 1) % self.tau_check == 0:
                # Causes an implicit sync point
                if not active.any():
                    break

            i += 1

        return final_adversarials
