import advertorch
import numpy as np
import torch

class RandomNoiseAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, targeted, evade_detector, eps, count=100, early_rejection_threshold=None, clip_min=0, clip_max=1):
        super().__init__(predict, None, clip_min, clip_max)
        self.targeted = targeted
        self.evade_detector = evade_detector
        self.eps = eps
        self.count = count
        self.early_rejection_threshold = early_rejection_threshold

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

        for _ in range(self.count):
            noise = torch.random.uniform(x.shape, device=x.device) * self.eps
            adversarials = torch.clamp(x + noise, min=self.clip_min, max=self.clip_max)
            
            outputs = self.predict(adversarials)

            successful = self.successful(outputs, y[active])

            distances = torch.max(torch.abs(x[active] - adversarials), dim=1)[0]
            better_distance = distances < best_distances[active]

            advertorch.utils.replace_active(...)#TODO: Completare

            if self.early_rejection_threshold is not None:
                reject = utils.early_rejection(x[active], adversarials, y[active], ...) # TODO

                # TODO: Altrove, ho anche filtrato con successful?
                active[active] = ~reject
