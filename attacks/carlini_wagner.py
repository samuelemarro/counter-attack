import advertorch.attacks as attacks
from advertorch.utils import clamp, calc_l2distsq, replace_active, replicate_input, tanh_rescale, torch_arctanh, to_one_hot

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import utils

CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
EPS = 1e-6
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10

class ERCarliniWagnerL2Attack(attacks.CarliniWagnerL2Attack):
    def __init__(self, predict, num_classes,
                confidence=0, targeted=False, learning_rate=0.01,
                binary_search_steps=9, max_iterations=10000,
                abort_early=True, initial_const=1e-3,
                clip_min=0., clip_max=1., loss_fn=None, early_rejection_threshold=None):
        super().__init__(predict, num_classes, confidence, targeted,
                        learning_rate, binary_search_steps, max_iterations, abort_early,
                        initial_const, clip_min, clip_max)

        self.early_rejection_threshold = early_rejection_threshold

    def _update_if_smaller_dist_succeed(
            self, adv_img, labs, output, l2distsq,
            cur_l2distsqs, cur_labels,
            final_l2distsqs, final_labels, final_advs, active):

        target_label = labs[active]
        output_logits = output
        _, output_label = torch.max(output_logits, 1)

        mask = (l2distsq < cur_l2distsqs[active]) & self._is_successful(
            output_logits, target_label, True)

        #cur_l2distsqs[mask] = l2distsq[mask]  # redundant
        replace_active(l2distsq, cur_l2distsqs, active, mask)

        #cur_labels[mask] = output_label[mask]
        replace_active(output_label, cur_labels, active, mask)

        mask = (l2distsq < final_l2distsqs[active]) & self._is_successful(
            output_logits, target_label, True)

        
        #final_l2distsqs[mask] = l2distsq[mask]
        replace_active(l2distsq, final_l2distsqs, active, mask)

        #final_labels[mask] = output_label[mask]
        replace_active(output_label, final_labels, active, mask)

        #final_advs[mask] = adv_img[mask]
        replace_active(adv_img, final_advs, active, mask)

    def _update_loss_coeffs(
            self, labs, cur_labels, batch_size, loss_coeffs,
            coeff_upper_bound, coeff_lower_bound, active):
        if self.targeted:
            successful = torch.eq(cur_labels[active], labs[active])
        else:
            successful = ~torch.eq(cur_labels[active], labs[active])

        successful = successful & ~torch.eq(cur_labels[active], INVALID_LABEL)

        #coeff_upper_bound[successful] = torch.min(loss_coeffs, coeff_upper_bound)[successful]
        replace_active(torch.min(loss_coeffs, coeff_upper_bound)[active], coeff_upper_bound, active, successful)

        #coeff_lower_bound[~successful] = torch.max(loss_coeffs, coeff_lower_bound)[~successful]
        replace_active(torch.max(loss_coeffs, coeff_lower_bound)[active], coeff_lower_bound, active, ~successful)

        coeff_check = coeff_upper_bound[active] < UPPER_CHECK

        #loss_coeffs[coeff_check] = (coeff_lower_bound[coeff_check] + coeff_upper_bound[coeff_check]) / 2
        replace_active(((coeff_lower_bound + coeff_upper_bound) / 2)[active], loss_coeffs, active, coeff_check)

        multiply_coeff = ~coeff_check & ~successful
        #loss_coeffs[multiply_coeff] *= 10
        replace_active(loss_coeffs[active] * 10, loss_coeffs, active, multiply_coeff)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)
        
        x = replicate_input(x)
        batch_size = len(x)
        coeff_lower_bound = x.new_zeros(batch_size)
        coeff_upper_bound = x.new_ones(batch_size) * CARLINI_COEFF_UPPER
        loss_coeffs = torch.ones_like(y).float() * self.initial_const
        final_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_advs = x.clone()
        x_atanh = self._get_arctanh_x(x)
        y_onehot = to_one_hot(y, self.num_classes).float()

        final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(x.device)
        final_labels = torch.LongTensor(final_labels).to(x.device)

        active = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

        # Start binary search
        for outer_step in range(self.binary_search_steps):
            delta = nn.Parameter(torch.zeros_like(x))
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
            cur_labels = [INVALID_LABEL] * batch_size
            cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(x.device)
            cur_labels = torch.LongTensor(cur_labels).to(x.device)
            prev_avgloss = PREV_LOSS_INIT

            if (self.repeat and outer_step == (self.binary_search_steps - 1)):
                loss_coeffs = coeff_upper_bound
            for ii in range(self.max_iterations):
                optimizer.zero_grad()
                adv_img = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)[active]
                transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)[active]
                output = self.predict(adv_img)

                l2distsq = calc_l2distsq(adv_img, transimgs_rescale)

                self._update_if_smaller_dist_succeed(
                    adv_img.detach(), y, output.detach(), l2distsq.detach(),
                    cur_l2distsqs, cur_labels,
                    final_l2distsqs, final_labels, final_advs, active)

                # Remove rejected examples
                if self.early_rejection_threshold is not None:
                    rejected = utils.early_rejection(x[active], adv_img.detach(), y[active], output.detach(), 2, self.early_rejection_threshold, self.targeted)

                    # This workaround avoids modifying "active"
                    # in-place, which would mess with
                    # gradient computation in backwards()
                    active_clone = active.clone()
                    active_clone[active] = ~rejected
                    active = active_clone

                    adv_img = adv_img[~rejected]
                    transimgs_rescale = transimgs_rescale[~rejected]
                    output = output[~rejected]

                if not active.any():
                    break

                loss = self._loss_fn(output, y_onehot[active], l2distsq, loss_coeffs[active])
                loss.backward()
                optimizer.step()

                if self.abort_early:
                    if ii % (self.max_iterations // NUM_CHECKS or 1) == 0:
                        avgloss = (loss / len(torch.nonzero(active))).detach()
                        if avgloss > prev_avgloss * ONE_MINUS_EPS:
                            break
                        prev_avgloss = avgloss

            self._update_loss_coeffs(
                y, cur_labels, batch_size,
                loss_coeffs, coeff_upper_bound, coeff_lower_bound, active)

            if not active.any():
                break

        return final_advs



class ERCarliniWagnerLinfAttack(attacks.CarliniWagnerLinfAttack):
    """
    The Carlini and Wagner LInfinity Attack, https://arxiv.org/abs/1608.04644

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
    :param early_rejection_threshold: the maximum distortion required for
        early rejection.
    """

    def __init__(self, predict, num_classes, min_tau=1 / 256,
                 initial_tau=1, tau_factor=0.9, initial_const=1e-5,
                 max_const=20, const_factor=2, reduce_const=False,
                 warm_start=True, targeted=False, learning_rate=5e-3,
                 max_iterations=1000, abort_early=True, clip_min=0.,
                 clip_max=1., loss_fn=None, return_best=True, early_rejection_threshold=None):
        """Carlini Wagner LInfinity Attack implementation in pytorch."""
        if loss_fn is not None:
            import warnings
            warnings.warn(
                "This Attack currently does not support a different loss"
                " function other than the default. Setting loss_fn manually"
                " is not effective."
            )

        loss_fn = None

        super().__init__(
            predict,
            num_classes,
            min_tau=min_tau,
            initial_tau=initial_tau,
            tau_factor=tau_factor,
            initial_const=initial_const,
            max_const=max_const,
            const_factor=const_factor,
            reduce_const=reduce_const,
            warm_start=warm_start,
            targeted=targeted,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            abort_early=abort_early,
            clip_min=clip_min,
            clip_max=clip_max,
            loss_fn=loss_fn,
            return_best=return_best
            )

        self.early_rejection_threshold = early_rejection_threshold

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
                    x[active],
                    modifiers[active],
                    starting_atanh[active],
                    y[active],
                    const,
                    taus[active])

                adversarials = tanh_rescale(
                    starting_atanh + modifiers,
                    self.clip_min,
                    self.clip_max).detach()

                successful = self._successful(outputs, y[active])

                if self.return_best:
                    distances = torch.max(
                                torch.abs(
                                    x[active] - adversarials[active]
                                    ).flatten(1),
                                dim=1)[0]
                    better_distance = distances < best_distances[active]

                    replace_active(adversarials[active],
                                   best_adversarials,
                                   active,
                                   successful & better_distance)
                    replace_active(distances,
                                   best_distances,
                                   active,
                                   successful & better_distance)
                else:
                    best_adversarials[active] = adversarials[active]


                drop = torch.zeros((len(torch.nonzero(active)),), dtype=torch.bool, device=x.device)

                # If early aborting is enabled, drop successful
                # samples with a small loss (the current adversarials
                # are saved regardless of whether they are dropped)
                if self.abort_early:
                    small_loss = loss < 0.0001 * const

                    drop = drop | (successful & small_loss)

                if self.early_rejection_threshold is not None:
                    reject = utils.early_rejection(x[active],
                                                   adversarials[active],
                                                   y[active],
                                                   outputs,
                                                   np.inf,
                                                   self.early_rejection_threshold,
                                                   self.targeted)

                    drop = drop | reject

                # This workaround avoids modifying "active"
                # in-place, which would mess with
                # gradient computation in backwards()
                active_clone = active.clone()
                active_clone[active] = ~drop
                active = active_clone

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

            replace_active(linf_distances,
                           taus,
                           active,
                           linf_lower & successful)

            # Save the remaining adversarials
            if self.return_best:
                better_distance = linf_distances < best_distances[active]
                replace_active(adversarials,
                               best_adversarials,
                               active,
                               successful & better_distance)
                replace_active(linf_distances,
                               best_distances,
                               active,
                               successful & better_distance)
            else:
                replace_active(adversarials,
                               best_adversarials,
                               active,
                               successful)

            taus *= self.tau_factor

            if self.reduce_const:
                initial_const /= 2

            # Drop failed samples or with a low tau
            low_tau = taus[active] <= self.min_tau
            drop = low_tau | (~successful)

            # If early rejection is enabled, drop the relevant samples
            if self.early_rejection_threshold is not None:
                reject = utils.early_rejection(x[active],
                                               adversarials,
                                               y[active],
                                               adversarial_outputs,
                                               np.inf,
                                               self.early_rejection_threshold,
                                               self.targeted)

                drop = drop | reject

            active[active] = ~drop

        return best_adversarials
