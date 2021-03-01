import advertorch.attacks as attacks
from advertorch.utils import clamp, calc_l2distsq, replicate_input, tanh_rescale, torch_arctanh, to_one_hot

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import utils

ONE_MINUS_EPS = 0.999999
TARGET_MULT = 10000.0

def get_carlini_linf_attack(target_model, num_classes, cuda_optimized=True, **kwargs):
    if cuda_optimized:
        return CarliniWagnerCUDALinfAttack(target_model, num_classes, **kwargs)
    else:
        return CarliniWagnerCPULinfAttack(target_model, num_classes, **kwargs)


class CarliniWagnerCPULinfAttack(attacks.Attack, attacks.LabelMixin):
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
    :param loss_fn: loss function
    :param return_best: if True, return the best adversarial found, else
        return the the last adversarial found.
    """

    def __init__(self, predict, num_classes, min_tau=1 / 256,
                 initial_tau=1, tau_factor=0.9, initial_const=1e-5,
                 max_const=20, const_factor=2, reduce_const=False,
                 warm_start=True, targeted=False, learning_rate=5e-3,
                 max_iterations=1000, abort_early=True, clip_min=0.,
                 clip_max=1., loss_fn=None, return_best=True):
        """Carlini Wagner LInfinity Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently does not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

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
        self.return_best = return_best

    def _get_arctanh_x(self, x):
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=0., max=1.) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def _outputs_and_loss(self, x, modifiers, starting_atanh, y, const, taus):
        adversarials = tanh_rescale(
            starting_atanh + modifiers, self.clip_min, self.clip_max)

        outputs = self.predict(adversarials)
        y_onehot = to_one_hot(y, self.num_classes).float()

        real = (y_onehot * outputs).sum(dim=1)

        other = ((1.0 - y_onehot) * outputs - (y_onehot * TARGET_MULT)
                 ).max(dim=1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.targeted:
            loss1 = torch.clamp(other - real, min=0.)
        else:
            loss1 = torch.clamp(real - other, min=0.)

        loss1 = const * loss1

        image_dimensions = tuple(range(1, len(x.shape)))
        taus_shape = (-1,) + (1,) * (len(x.shape) - 1)

        penalties = torch.clamp(
            torch.abs(x - adversarials) - taus.view(taus_shape), min=0)
        loss2 = torch.sum(penalties, dim=image_dimensions)

        assert loss1.shape == loss2.shape

        loss = loss1 + loss2
        return outputs.detach(), loss

    def _successful(self, outputs, y):
        adversarial_labels = torch.argmax(outputs, dim=1)

        if self.targeted:
            return torch.eq(adversarial_labels, y)
        else:
            return ~torch.eq(adversarial_labels, y)

    def _run_attack(self, x, y, initial_const, taus, prev_adversarials):
        assert len(x) == len(taus)
        batch_size = len(x)
        best_adversarials = x.clone().detach()
        best_distances = torch.ones((batch_size,),
                                    device=x.device) * float("inf")

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

        while torch.any(active) and const < self.max_const:
            for _ in range(self.max_iterations):
                optimizer.zero_grad()
                outputs, loss = self._outputs_and_loss(
                    x,
                    modifiers,
                    starting_atanh,
                    y,
                    const,
                    taus)

                adversarials = tanh_rescale(
                    starting_atanh + modifiers,
                    self.clip_min,
                    self.clip_max).detach()

                successful = self._successful(outputs, y)

                if self.return_best:
                    distances = torch.max(
                        torch.abs(
                            x - adversarials
                        ).flatten(1),
                        dim=1)[0]
                    better_distance = distances < best_distances

                    best_adversarials = utils.fast_boolean_choice(
                        best_adversarials, adversarials, successful & better_distance)
                else:
                    best_adversarials = adversarials

                # If early aborting is enabled, drop successful
                # samples with a small loss (the current adversarials
                # are saved regardless of whether they are dropped)
                if self.abort_early:
                    small_loss = loss < 0.0001 * const

                    active = active & ~(successful & small_loss)

                    if not active.any():
                        break

                # Update the modifiers
                total_loss = torch.sum(loss)
                total_loss.backward()
                optimizer.step()

            # Give more weight to the output loss
            const *= self.const_factor

        return best_adversarials

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        x = replicate_input(x)
        batch_size = len(x)
        best_adversarials = x.clone()
        best_distances = torch.ones((batch_size,),
                                    device=x.device) * float("inf")

        # An array of booleans that stores which samples have not converged
        # yet
        active = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

        initial_const = self.initial_const
        taus = torch.ones((batch_size,), device=x.device) * self.initial_tau

        # The previous adversarials. This is used to perform a "warm start"
        # during optimisation
        prev_adversarials = x.clone()

        while torch.any(active):
            adversarials = self._run_attack(
                x[active],
                y[active],
                initial_const,
                taus[active],
                prev_adversarials[active].clone())

            # Store the adversarials for the next iteration,
            # even if they failed
            prev_adversarials[active] = adversarials

            adversarial_outputs = self.predict(adversarials)
            successful = self._successful(adversarial_outputs, y[active])

            # If the Linf distance is lower than tau and the adversarial
            # is successful, use it as the new tau
            linf_distances = torch.max(
                torch.abs(adversarials - x[active]).flatten(1),
                dim=1)[0]
            linf_lower = linf_distances < taus[active]

            utils.replace_active(linf_distances,
                                 taus,
                                 active,
                                 linf_lower & successful)

            # Save the remaining adversarials
            if self.return_best:
                better_distance = linf_distances < best_distances[active]
                utils.replace_active(adversarials,
                                     best_adversarials,
                                     active,
                                     successful & better_distance)
                utils.replace_active(linf_distances,
                                     best_distances,
                                     active,
                                     successful & better_distance)
            else:
                utils.replace_active(adversarials,
                                     best_adversarials,
                                     active,
                                     successful)

            taus *= self.tau_factor

            if self.reduce_const:
                initial_const /= 2

            # Drop failed samples or with a low tau
            low_tau = taus[active] <= self.min_tau
            drop = low_tau | (~successful)
            active[active] = ~drop

        return best_adversarials


class CarliniWagnerCUDALinfAttack(attacks.Attack, attacks.LabelMixin):
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
    :param loss_fn: loss function
    :param return_best: if True, return the best adversarial found, else
        return the the last adversarial found.
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
        been declared inactive. Defaults to the value of return_best.
    """

    def __init__(self, predict, num_classes, min_tau=1 / 256,
                 initial_tau=1, tau_factor=0.9, initial_const=1e-5,
                 max_const=20, const_factor=2, reduce_const=False,
                 warm_start=True, targeted=False, learning_rate=5e-3,
                 max_iterations=1000, abort_early=True, clip_min=0.,
                 clip_max=1., loss_fn=None, return_best=True,
                 tau_check=1, const_check=5, inner_check=500,
                 update_inactive=None):
        """Carlini Wagner LInfinity Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently does not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

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
        self.return_best = return_best
        
        self.tau_check = tau_check
        self.const_check = const_check
        self.inner_check = inner_check

        if update_inactive is None:
            update_inactive = return_best

        self.update_inactive = update_inactive

    def _get_arctanh_x(self, x):
        result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
                       min=0., max=1.) * 2 - 1
        return torch_arctanh(result * ONE_MINUS_EPS)

    def _outputs_and_loss(self, x, modifiers, starting_atanh, y, const, taus):
        adversarials = tanh_rescale(
            starting_atanh + modifiers, self.clip_min, self.clip_max)

        outputs = self.predict(adversarials)
        y_onehot = to_one_hot(y, self.num_classes).float()

        real = (y_onehot * outputs).sum(dim=1)

        other = ((1.0 - y_onehot) * outputs - (y_onehot * TARGET_MULT)
                 ).max(dim=1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        if self.targeted:
            loss1 = torch.clamp(other - real, min=0.)
        else:
            loss1 = torch.clamp(real - other, min=0.)

        loss1 = const * loss1

        image_dimensions = tuple(range(1, len(x.shape)))
        taus_shape = (-1,) + (1,) * (len(x.shape) - 1)

        penalties = torch.clamp(
            torch.abs(x - adversarials) - taus.view(taus_shape), min=0)
        loss2 = torch.sum(penalties, dim=image_dimensions)

        assert loss1.shape == loss2.shape

        loss = loss1 + loss2
        return outputs.detach(), loss

    def _successful(self, outputs, y):
        adversarial_labels = torch.argmax(outputs, dim=1)

        if self.targeted:
            return torch.eq(adversarial_labels, y)
        else:
            return ~torch.eq(adversarial_labels, y)

    def _run_attack(self, x, y, initial_const, taus, prev_adversarials):
        assert len(x) == len(taus)
        batch_size = len(x)
        best_adversarials = x.clone().detach()
        best_distances = torch.ones((batch_size,),
                                    device=x.device) * float("inf")

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

        j = 0
        stop_search = False

        while (not stop_search) and const < self.max_const:
            for k in range(self.max_iterations):
                optimizer.zero_grad()
                outputs, loss = self._outputs_and_loss(
                    x,
                    modifiers,
                    starting_atanh,
                    y,
                    const,
                    taus)

                adversarials = tanh_rescale(
                    starting_atanh + modifiers,
                    self.clip_min,
                    self.clip_max).detach()

                successful = self._successful(outputs, y)

                if self.return_best:
                    distances = torch.max(
                        torch.abs(
                            x - adversarials
                        ).flatten(1),
                        dim=1)[0]
                    better_distance = distances < best_distances
                    replace = successful & better_distance
                else:
                    replace = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

                if not self.update_inactive:
                    replace = replace & active

                best_adversarials = utils.fast_boolean_choice(best_adversarials, adversarials, replace)
                best_distances = utils.fast_boolean_choice(best_distances, distances, replace)

                # If early aborting is enabled, drop successful
                # samples with a small loss (the current adversarials
                # are saved regardless of whether they are dropped)
                if self.abort_early:
                    small_loss = loss < 0.0001 * const

                    active = active & ~(successful & small_loss)

                    if (k + 1) % self.inner_check == 0 and self.inner_check != 0:
                        # Causes an implicit sync point
                        if not active.any():
                            # Break from both loops
                            stop_search = True
                            break

                # Update the modifiers
                total_loss = torch.sum(loss)
                total_loss.backward()
                optimizer.step()

            if stop_search:
                break

            if self.abort_early and (j + 1) % self.const_check == 0 and self.const_check != 0:
                # Causes an implicit sync point
                if not active.any():
                    break

            # Give more weight to the output loss
            const *= self.const_factor

        return best_adversarials

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        x = replicate_input(x)
        batch_size = len(x)
        best_adversarials = x.clone()
        best_distances = torch.ones((batch_size,),
                                    device=x.device) * float("inf")

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
            adversarials = self._run_attack(
                x,
                y,
                initial_const,
                taus,
                prev_adversarials.clone())

            # Store the adversarials for the next iteration,
            # even if they failed
            prev_adversarials = adversarials

            adversarial_outputs = self.predict(adversarials)
            successful = self._successful(adversarial_outputs, y)

            # If the Linf distance is lower than tau and the adversarial
            # is successful, use it as the new tau
            linf_distances = torch.max(
                torch.abs(adversarials - x).flatten(1),
                dim=1)[0]
            linf_lower = linf_distances < taus

            taus = utils.fast_boolean_choice(taus, linf_distances, linf_lower & successful)

            # Save the remaining adversarials
            if self.return_best:
                better_distance = linf_distances < best_distances
                replace = successful & better_distance
            else:
                replace = successful
            
            if not self.update_inactive:
                replace = replace & active

            best_distances = utils.fast_boolean_choice(best_distances, linf_distances, replace)
            best_adversarials = utils.fast_boolean_choice(best_adversarials, adversarials, replace)

            taus *= self.tau_factor
            max_tau *= self.tau_factor

            if self.reduce_const:
                initial_const /= 2

            # Drop failed samples or with a low tau
            low_tau = taus <= self.min_tau
            drop = low_tau | (~successful)
            active = active & (~drop)

            if (i + 1) % self.tau_check == 0 and self.tau_check != 0:
                # Causes an implicit sync point
                if not active.any():
                    break

            i += 1

        return best_adversarials

