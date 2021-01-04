import copy

import advertorch
import numpy as np
import torch

import utils
# Nota: Aggiorna eps se ha una distanza più bassa (non solo se ha successo)


class EpsilonBinarySearchAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, evade_detector, ord, attack, unsqueeze, targeted=False, min_eps=0, max_eps=1, eps_initial_search_steps=9, eps_binary_search_steps=9):
        super().__init__(predict, None, None, None)

        self.predict = predict
        self.evade_detector = evade_detector
        self.ord = ord
        assert attack.targeted == targeted
        self.attack = attack
        self.unsqueeze = unsqueeze
        self.targeted = targeted
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps_initial_search_steps = eps_initial_search_steps
        self.eps_binary_search_steps = eps_binary_search_steps

    def perturb_standard(self, x, y, eps):
        assert len(x) == len(y)
        assert len(eps) == len(x)

        # Perform a shallow copy
        attack = copy.copy(self.attack)

        if self.unsqueeze:
            eps = eps.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        attack.eps = eps

        return attack(x, y=y)

    def successful(self, adversarial_outputs, y):
        adversarial_labels = torch.argmax(adversarial_outputs, dim=1)

        if self.targeted:
            return torch.eq(adversarial_labels, y)
        else:
            return ~torch.eq(adversarial_labels, y)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        best_adversarials = x.clone()
        N = x.shape[0]

        eps_lower_bound = torch.ones((N,), device=x.device) * self.min_eps
        eps_upper_bound = torch.ones((N,), device=x.device) * self.max_eps
        best_distances = torch.ones(
            (N), device=x.device) * torch.finfo(x.dtype).max

        initial_search_eps = eps_upper_bound.clone()
        for _ in range(self.eps_initial_search_steps):
            adversarials = self.perturb_standard(
                x, y, initial_search_eps).detach()
            adversarial_outputs = self.predict(adversarials).detach()
            successful = self.successful(adversarial_outputs, y)

            distances = utils.adversarial_distance(x, adversarials, self.ord)
            better_distances = distances < best_distances

            replace = successful & better_distances

            best_adversarials = utils.fast_boolean_choice(
                best_adversarials, adversarials, replace)
            best_distances = utils.fast_boolean_choice(
                best_distances, distances, replace)

            # Success: Reduce the upper bound
            eps_upper_bound = utils.fast_boolean_choice(
                eps_upper_bound, initial_search_eps, successful)

            # Halve eps, regardless of the success
            initial_search_eps = initial_search_eps / 2

        for _ in range(self.eps_binary_search_steps):
            eps = (eps_lower_bound + eps_upper_bound) / 2
            adversarials = self.perturb_standard(x, y, eps).detach()
            adversarial_outputs = self.predict(adversarials).detach()
            successful = self.successful(adversarial_outputs, y)

            distances = utils.adversarial_distance(x, adversarials, self.ord)
            better_distances = distances < best_distances
            replace = successful & better_distances

            best_adversarials = utils.fast_boolean_choice(
                best_adversarials, adversarials, replace)
            best_distances = utils.fast_boolean_choice(
                best_distances, distances, replace)

            # Success: Reduce the upper bound
            eps_upper_bound = utils.fast_boolean_choice(
                eps_upper_bound, eps, successful)

            # Failure: Increase the lower bound
            eps_lower_bound = utils.fast_boolean_choice(
                eps_lower_bound, eps, ~successful)

        assert best_adversarials.shape == x.shape

        return best_adversarials
