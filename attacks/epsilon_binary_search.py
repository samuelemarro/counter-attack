import copy

import advertorch
from advertorch.utils import replace_active
import numpy as np
import torch

import utils
# Nota: Aggiorna eps se ha una distanza più bassa (non solo se ha successo)

class EpsilonBinarySearchAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, evade_detector, ord, attack, unsqueeze, targeted=False, min_eps=0, max_eps=1, initial_search_steps=9, binary_search_steps=9,
                early_rejection_threshold=None):
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
        self.initial_search_steps = initial_search_steps
        self.binary_search_steps = binary_search_steps
        self.early_rejection_threshold = early_rejection_threshold

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

        active = torch.ones((N,), dtype=bool, device=x.device)

        eps_lower_bound = torch.ones((N,), device=x.device) * self.min_eps
        eps_upper_bound = torch.ones((N,), device=x.device) * self.max_eps
        last_distances = torch.ones((N), device=x.device) * np.inf

        initial_search_eps = eps_upper_bound.clone()
        for _ in range(self.initial_search_steps):
            if not active.any():
                break

            adversarials = self.perturb_standard(x[active], y[active], initial_search_eps[active]).detach()
            adversarial_outputs = self.predict(adversarials).detach()
            successful = self.successful(adversarial_outputs, y[active])

            distances = utils.adversarial_distance(x[active], adversarials, self.ord)
            better_distances = distances < last_distances[active]

            replace = successful & better_distances

            replace_active(adversarials, best_adversarials, active, replace)
            replace_active(distances, last_distances, active, replace)

            # Success: Reduce the upper bound
            replace_active(initial_search_eps, eps_upper_bound, active, replace)

            # Halve eps, regardless of the success
            initial_search_eps = initial_search_eps / 2

            if self.early_rejection_threshold is not None:
                reject = utils.early_rejection(x[active], adversarials, y[active],
                    adversarial_outputs, self.ord, self.early_rejection_threshold, self.targeted)

                active[active] = ~reject

        for _ in range(self.binary_search_steps):
            if not active.any():
                break

            eps = (eps_lower_bound[active] + eps_upper_bound[active]) / 2
            adversarials = self.perturb_standard(x[active], y[active], eps).detach()
            adversarial_outputs = self.predict(adversarials).detach()
            successful = self.successful(adversarial_outputs, y[active])

            distances = utils.adversarial_distance(x[active], adversarials, self.ord)
            better_distances = distances < last_distances[active]
            replace = successful & better_distances

            replace_active(adversarials, best_adversarials, active, replace)
            replace_active(distances, last_distances, active, replace)

            # Success: Reduce the upper bound
            replace_active(eps, eps_upper_bound, active, replace)

            # Failure: Increase the lower bound
            replace_active(eps, eps_lower_bound, active, ~replace)

            if self.early_rejection_threshold is not None:
                reject = utils.early_rejection(x[active], adversarials, y[active],
                    adversarial_outputs, self.ord, self.early_rejection_threshold, self.targeted)

                active[active] = ~reject

        assert best_adversarials.shape == x.shape

        return best_adversarials