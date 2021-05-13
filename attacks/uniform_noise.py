import advertorch
import torch

import utils

MAX_DISTANCE = 1e8

class UniformNoiseAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, p, targeted, eps=0.3, count=100, clip_min=0, clip_max=1):
        super().__init__(predict, None, clip_min, clip_max)
        self.p = p
        self.targeted = targeted
        self.eps = eps
        self.count = count

    def successful(self, adversarials, y):
        return utils.check_successful(self.predict, adversarials, y, self.targeted)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        batch_size = len(x)

        with torch.no_grad():
            best_adversarials = x.clone()
            best_distances = torch.ones((batch_size,), device=x.device) * MAX_DISTANCE

            for _ in range(self.count):
                noise = torch.rand(x.shape, device=x.device)

                scaled_noise = (noise * 2 - 1) * self.eps
                adversarials = torch.clamp(
                    x + scaled_noise, min=self.clip_min, max=self.clip_max)

                assert adversarials.shape == x.shape

                successful = self.successful(adversarials, y).detach()

                distances = utils.adversarial_distance(
                    x, adversarials, self.p)
                better_distance = distances < best_distances

                assert len(better_distance) == len(best_adversarials) == len(best_distances)

                best_adversarials = utils.fast_boolean_choice(
                    best_adversarials, adversarials, successful & better_distance)
                best_distances = utils.fast_boolean_choice(
                    best_distances, distances, successful & better_distance)

            return best_adversarials
