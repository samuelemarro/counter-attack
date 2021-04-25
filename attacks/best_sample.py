import logging

import advertorch.attacks as attacks
import torch
import torch.nn as nn

import utils

logger = logging.getLogger(__name__)


class BestSampleTracker:
    def __init__(self, genuines, labels, p, targeted):
        assert len(genuines) == len(labels)

        genuines = genuines.detach()
        labels = labels.detach()
        self.genuines = genuines
        self.labels = labels
        self.p = p
        self.targeted = targeted
        self.best = torch.zeros_like(genuines)
        self.found_adversarial = torch.zeros(
            [len(genuines)], device=genuines.device, dtype=torch.bool)
        self.best_distances = torch.zeros(
            [len(genuines)], device=genuines.device, dtype=genuines.dtype)


class BestSampleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.training = model.training
        self.tracker = None

    def forward(self, x):
        if self.tracker is None:
            raise RuntimeError('No best sample tracker set.')

        assert len(x) == len(self.tracker.genuines)

        # Don't detach here: attacks might require the gradients
        outputs = self.model(x)

        with torch.no_grad():
            adversarial_labels = torch.argmax(outputs, dim=1)
            if self.tracker.targeted:
                successful = torch.eq(adversarial_labels, self.tracker.labels)
            else:
                successful = ~torch.eq(adversarial_labels, self.tracker.labels)

            distances = utils.adversarial_distance(
                self.tracker.genuines, x, self.tracker.p)
            better_distance = distances < self.tracker.best_distances

            # Replace only if successful and with a better distance
            replace = successful & (better_distance | (
                ~self.tracker.found_adversarial))

            self.tracker.best = utils.fast_boolean_choice(
                self.tracker.best, x, replace)
            self.tracker.best_distances = utils.fast_boolean_choice(
                self.tracker.best_distances, distances, replace)
            self.tracker.found_adversarial = self.tracker.found_adversarial | replace

        return outputs


class BestSampleAttack(attacks.Attack, attacks.LabelMixin):
    def __init__(self, wrapped_model, inner_attack, p, targeted, suppress_warning=False):
        if not suppress_warning:
            if wrapped_model is not inner_attack.predict:
                logger.warning('BestSampleAttack was passed a model that is different from inner_attack\'s model.')

        self.wrapped_model = wrapped_model
        self.inner_attack = inner_attack
        self.p = p
        self.targeted = targeted

    def perturb(self, x, y=None, **kwargs):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        # Create a tracker that will be only used for this batch
        tracker = BestSampleTracker(x, y, self.p, self.targeted)
        self.wrapped_model.tracker = tracker

        last_adversarials = self.inner_attack(x, y=y, **kwargs)

        # In case the last adversarials were not tested
        self.wrapped_model(last_adversarials)

        # If the wrapper failed to find some adversarials, use the
        # last ones
        final_adversarials = utils.fast_boolean_choice(
            last_adversarials, tracker.best, tracker.found_adversarial)

        self.wrapped_model.tracker = None

        return final_adversarials
