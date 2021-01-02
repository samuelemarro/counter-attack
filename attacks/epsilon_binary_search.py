import copy

import advertorch
import numpy as np
import torch

import utils
# Nota: Aggiorna eps se ha una distanza più bassa (non solo se ha successo)

# TODO: Passare a fast_boolean_choice

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

        active = torch.ones((N,), dtype=bool, device=x.device)

        eps_lower_bound = torch.ones((N,), device=x.device) * self.min_eps
        eps_upper_bound = torch.ones((N,), device=x.device) * self.max_eps
        last_distances = torch.ones((N), device=x.device) * np.inf

        initial_search_eps = eps_upper_bound.clone()
        for _ in range(self.eps_initial_search_steps):
            if not active.any():
                break

            adversarials = self.perturb_standard(x[active], y[active], initial_search_eps[active]).detach()
            adversarial_outputs = self.predict(adversarials).detach()
            successful = self.successful(adversarial_outputs, y[active])

            distances = utils.adversarial_distance(x[active], adversarials, self.ord)
            better_distances = distances < last_distances[active]

            replace = successful & better_distances

            utils.replace_active(adversarials, best_adversarials, active, replace)
            utils.replace_active(distances, last_distances, active, replace)

            # Success: Reduce the upper bound
            utils.replace_active(initial_search_eps[active], eps_upper_bound, active, replace)

            # Halve eps, regardless of the success
            initial_search_eps = initial_search_eps / 2

        for _ in range(self.eps_binary_search_steps):
            if not active.any():
                break

            eps = (eps_lower_bound[active] + eps_upper_bound[active]) / 2
            adversarials = self.perturb_standard(x[active], y[active], eps).detach()
            adversarial_outputs = self.predict(adversarials).detach()
            successful = self.successful(adversarial_outputs, y[active])

            distances = utils.adversarial_distance(x[active], adversarials, self.ord)
            better_distances = distances < last_distances[active]
            replace = successful & better_distances

            utils.replace_active(adversarials, best_adversarials, active, replace)
            utils.replace_active(distances, last_distances, active, replace)

            # Success: Reduce the upper bound
            utils.replace_active(eps, eps_upper_bound, active, replace)

            # Failure: Increase the lower bound
            utils.replace_active(eps, eps_lower_bound, active, ~replace)

        assert best_adversarials.shape == x.shape

        return best_adversarials