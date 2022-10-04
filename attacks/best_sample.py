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
    
    def successful(self, outputs, relevant_labels):
        adversarial_labels = torch.argmax(outputs, dim=1)
        if self.tracker.targeted:
            return torch.eq(adversarial_labels, relevant_labels)
        else:
            return ~torch.eq(adversarial_labels, relevant_labels)

    def forward(self, x, active_mask=None, filter_=None):
        if self.tracker is None:
            raise RuntimeError('No best sample tracker set.')

        # Don't detach here: attacks might require the gradients
        outputs = self.model(x)

        relevant_labels = self.tracker.labels
        relevant_genuines = self.tracker.genuines
        relevant_best_distances = self.tracker.best_distances
        relevant_found_adversarial = self.tracker.found_adversarial

        if active_mask is None:
            assert len(x) == len(self.tracker.genuines)
        else:
            assert len(x) == torch.count_nonzero(active_mask)
            # Boolean indexing causes a CUDA sync, which is why we do it only
            # if absolutely necessary
            relevant_labels = relevant_labels[active_mask]
            relevant_genuines = relevant_genuines[active_mask]
            relevant_best_distances = relevant_best_distances[active_mask]
            relevant_found_adversarial = relevant_found_adversarial[active_mask]

            # x doesn't need to be masked, since len(x) == torch.count_nonzero(active_mask)

        with torch.no_grad():
            successful = self.successful(outputs, relevant_labels)

            distances = utils.adversarial_distance(
                relevant_genuines, x, self.tracker.p)
            better_distance = distances < relevant_best_distances

            # Replace only if successful and with a better distance
            replace = successful & (better_distance | (
                ~relevant_found_adversarial))

            # filter_ restricts updates to only some samples
            if filter_ is not None:
                replace &= filter_

            new_found_adversarial = relevant_found_adversarial | replace

            if active_mask is None:
                self.tracker.best = utils.fast_boolean_choice(
                    self.tracker.best, x, replace)
                self.tracker.best_distances = utils.fast_boolean_choice(
                    self.tracker.best_distances, distances, replace)
                self.tracker.found_adversarial = new_found_adversarial
            else:
                # A masked replacement requires a different function
                utils.replace_active(x, self.tracker.best, active_mask, replace)
                utils.replace_active(distances, self.tracker.best_distances, active_mask, replace)
                self.tracker.found_adversarial[active_mask] = new_found_adversarial

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
        self.predict = wrapped_model.model

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
