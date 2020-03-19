import advertorch
import numpy as np
import torch

import utils

# TODO: Manual seed globale da console?

class UniformNoiseAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, p, targeted, eps=0.3, count=100, early_rejection_threshold=None, clip_min=0, clip_max=1, stochastic_consistency=False):
        super().__init__(predict, None, clip_min, clip_max)
        self.p = p
        self.targeted = targeted
        self.eps = eps
        self.count = count
        self.early_rejection_threshold = early_rejection_threshold
        self.stochastic_consistency = stochastic_consistency

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

        batch_size = len(x)

        best_adversarials = x.clone()
        best_distances = torch.ones((batch_size,), device=x.device) * np.inf
        active = torch.ones((batch_size,), dtype=bool, device=x.device)

        if self.stochastic_consistency:
            tensor_ids = torch.arange(len(x), device=x.device)
            generator = utils.ConsistentGenerator(
                lambda:
                torch.rand(x.shape[1:], device=x.device)
            )

        for _ in range(self.count):
            if self.stochastic_consistency:
                noise = generator.batch_generate(tensor_ids[active].cpu(), x[active])
            else:
                noise = torch.rand(x.shape, device=x.device)

            scaled_noise = (noise * 2 - 1) * self.eps
            adversarials = torch.clamp(x + scaled_noise, min=self.clip_min, max=self.clip_max)[active]
            
            outputs = self.predict(adversarials)

            successful = self.successful(outputs, y[active])

            distances = utils.adversarial_distance(x[active], adversarials, self.p)
            better_distance = distances < best_distances[active]

            advertorch.utils.replace_active(adversarials.detach(), best_adversarials, active, successful & better_distance)
            advertorch.utils.replace_active(distances.detach(), best_distances, active, successful & better_distance)

            if self.early_rejection_threshold is not None:
                reject = utils.early_rejection(x[active], adversarials, y[active], outputs, self.p, self.early_rejection_threshold, self.targeted)

                active[active] = ~reject

            if not active.any():
                break

        return best_adversarials
