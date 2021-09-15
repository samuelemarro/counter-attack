import torch

import attacks
import utils

MAX_DISTANCE = 1e8

# N.B.: This implementation updates eps based only on whether the attack was successful.
# Whether the attack found a better distance does not influence the eps update.

class EpsilonBinarySearchAttack(attacks.AdvertorchWrapper):
    def __init__(self, inner_attack, p, targeted=False, min_eps=0, max_eps=1, eps_initial_search_steps=9, eps_initial_search_factor=0.5, eps_binary_search_steps=9):
        if not isinstance(inner_attack, attacks.EpsilonAttack):
            raise ValueError('inner_attack must be an EpsilonAttack.')

        super().__init__(inner_attack)

        self.p = p
        assert inner_attack.targeted == targeted
        self.targeted = targeted
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps_initial_search_steps = eps_initial_search_steps
        self.eps_initial_search_factor = eps_initial_search_factor
        self.eps_binary_search_steps = eps_binary_search_steps

    def perturb_standard(self, x, y, eps):
        assert len(x) == len(y)
        return self.inner_attack(x, y=y, eps=eps)

    def successful(self, adversarials, y):
        return utils.check_successful(self.predict, adversarials, y, self.targeted)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        best_adversarials = x.clone()
        batch_size = len(x)

        eps_lower_bound = torch.ones((batch_size,), device=x.device) * self.min_eps
        eps_upper_bound = torch.ones((batch_size,), device=x.device) * self.max_eps
        best_distances = torch.ones(
            (batch_size), device=x.device) * MAX_DISTANCE

        initial_search_eps = eps_upper_bound.clone()
        for _ in range(self.eps_initial_search_steps):
            adversarials = self.perturb_standard(
                x, y, initial_search_eps).detach()

            assert adversarials.shape == x.shape

            successful = self.successful(adversarials, y).detach()

            distances = utils.adversarial_distance(x, adversarials, self.p)
            better_distance = distances < best_distances

            replace = successful & better_distance

            best_adversarials = utils.fast_boolean_choice(
                best_adversarials, adversarials, replace)
            best_distances = utils.fast_boolean_choice(
                best_distances, distances, replace)

            # Success: Reduce the upper bound
            eps_upper_bound = utils.fast_boolean_choice(
                eps_upper_bound, initial_search_eps, successful)

            # Reduce eps, regardless of the success
            initial_search_eps = initial_search_eps * self.eps_initial_search_factor

        for _ in range(self.eps_binary_search_steps):
            eps = (eps_lower_bound + eps_upper_bound) / 2
            adversarials = self.perturb_standard(x, y, eps).detach()
            successful = self.successful(adversarials, y).detach()

            distances = utils.adversarial_distance(x, adversarials, self.p)
            better_distance = distances < best_distances
            replace = successful & better_distance

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
