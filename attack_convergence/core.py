import json
import logging

import advertorch.attacks as attacks
import click
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn

import parsing
import utils

logger = logging.getLogger(__name__)

class AttackStopException(Exception):
    pass

class ConvergenceTracker:
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
        self.counter = 0
        self.stats = []


class ConvergenceWrapper(nn.Module):
    def __init__(self, model, track_every, stop=None):
        super().__init__()
        self.model = model
        self.track_every = track_every
        self.training = model.training
        self.tracker = None
        self.stop = stop

    def forward(self, x, active_mask=None, filter_=None, force_track=False):
        if self.tracker is None:
            raise RuntimeError('No convergence tracker set.')

        self.tracker.counter += 1

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
            adversarial_labels = torch.argmax(outputs, dim=1)
            if self.tracker.targeted:
                successful = torch.eq(adversarial_labels, relevant_labels)
            else:
                successful = ~torch.eq(adversarial_labels, relevant_labels)

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

            if force_track or self.tracker.counter >= self.stop or self.tracker.counter == 1 or (self.tracker.counter % self.track_every == 0):
                # print(self.tracker.counter % self.track_every)
                self.tracker.stats.append((self.tracker.counter, self.tracker.found_adversarial.detach().cpu(), self.tracker.best_distances.detach().cpu()))

            if self.tracker.counter >= self.stop:
                raise AttackStopException()

        return outputs


class ConvergenceAttack(attacks.Attack, attacks.LabelMixin):
    def __init__(self, wrapped_model, inner_attack, p, targeted, suppress_warning=False):
        if not suppress_warning:
            if wrapped_model is not inner_attack.predict:
                logger.warning('ConvergenceAttack was passed a model that is different from inner_attack\'s model.')

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
        tracker = ConvergenceTracker(x, y, self.p, self.targeted)
        self.wrapped_model.tracker = tracker

        try:
            last_adversarials = self.inner_attack(x, y=y, **kwargs)

            # Run the last adversarials in case they were not tested
            # force_track=True ensures that we know the final distance
            self.wrapped_model(last_adversarials, force_track=True)
        except AttackStopException:
            pass

        stats = list(tracker.stats)
        final_stats = tracker.stats[-1]

        self.wrapped_model.tracker = None

        return stats, final_stats

class CustomIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        assert len(indices) <= len(dataset)
        assert all(i >= 0 for i in indices)
        assert max(indices) < len(dataset)
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.indices[idx], self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
