import copy

import advertorch
import torch
import torch.nn as nn

from advertorch.utils import is_float_or_torch_tensor
import advertorch.attacks.iterative_projected_gradient as ipg

import numpy as np

import utils

MAX_DISTANCE = 1e10

# TODO: Remember to always check the .targeted of what you're working with,
# as well as if you're using the standard or defended model

class RandomTargetEvasionAttack:
    """Chooses a random label that is not the
    original one.
    """
    def __init__(self, classifier, attack_on_detector_classifier):
        self.classifier = classifier
        assert attack_on_detector_classifier.targeted
        self.attack_on_detector_classifier = attack_on_detector_classifier
        self.targeted = False # Always false

    def perturb(self, x, y=None, **kwargs):
        predictions = self.classifier(x)

        assert len(predictions) == len(x)

        num_classes = predictions.shape[1]

        original_labels = torch.argmax(predictions, axis=-1)

        target_labels = []

        for original_label in original_labels:
            target_label = None

            while target_label is None or target_label == original_label:
                target_label = torch.randint(0, num_classes, (1,))

            target_labels.append(target_label)

        target_labels = torch.cat(target_labels).to(original_labels.device)

        assert torch.all(torch.logical_not(torch.eq(original_labels, target_labels)))

        return self.attack_on_detector_classifier.perturb(x, y=target_labels, **kwargs)


class TopKEvasionAttack:
    """Chooses the kth top predicted label (by default the 2nd).
    """
    def __init__(self, classifier, attack_on_detector_classifier, k=2):
        self.classifier = classifier
        assert attack_on_detector_classifier.targeted
        self.attack_on_detector_classifier = attack_on_detector_classifier
        self.k = k
        self.targeted = False # Always false

    def perturb(self, x, y=None, **kwargs):
        predictions = self.classifier(x)
        _, target_labels = torch.topk(predictions, k=self.k)
        target_labels = target_labels[:, -1]

        assert torch.all(torch.logical_not(torch.eq(torch.argmax(predictions, axis=-1), target_labels)))

        return self.attack_on_detector_classifier.perturb(x, y=target_labels, **kwargs)

# TODO: Usa check_success con has_detector=False


# Nota: In ogni attacco, per "predict" si intende il modello indifeso

class AttackPool(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, attacks, p):
        self.predict = predict
        assert all(not attack.targeted for attack in attacks)
        self.attacks = attacks
        self.p = p
        self.targeted = False # Always false

    def successful(self, adversarials, y):
        return utils.check_success(self.predict, adversarials, y, False)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)
        
        pool_results = torch.stack([attack.perturb(x, y=y) for attack in self.attacks], 1)

        assert pool_results.shape[1] == len(self.attacks)
        assert len(pool_results) == len(x)
        
        best_adversarials = []

        for original, pool_result, label in zip(x, pool_results, y):
            expanded_label = label.expand(len(pool_result))
            
            successful = self.successful(pool_result, expanded_label)
            assert successful.shape == (len(successful),)

            if successful.any():
                successful_adversarials = pool_result[successful]

                # Bisogna prima appiattire affinché pairwise_distance calcoli correttamente
                # (original è singolo, pool_result è una batch)
                distances = torch.pairwise_distance(original.flatten(), pool_result.flatten(1), self.p)

                assert distances.shape == (len(successful_adversarials),)

                best_distance_index = torch.argmin(distances)
            
                best_adversarials.append(successful_adversarials[best_distance_index])
            else:
                # All the attacks failed: Return the original
                best_adversarials.append(original)

            

        best_adversarials = torch.stack(best_adversarials)

        assert best_adversarials.shape == x.shape

        return best_adversarials

# TODO: Usa check_success con has_detector=False

# Nota: Aggiorna eps se ha una distanza più bassa

class EpsilonBinarySearchAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, ord, attack, unsqueeze, targeted=False, min_eps=0, max_eps=1, initial_search_steps=9, binary_search_steps=9):
        super().__init__(predict, None, None, None)

        self.predict = predict
        self.ord = ord
        self.attack = attack
        self.unsqueeze = unsqueeze
        self.targeted = targeted
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.initial_search_steps = initial_search_steps
        self.binary_search_steps = binary_search_steps

    def perturb_standard(self, x, y, eps):
        assert len(x) == len(y)
        assert len(eps) == len(x)

        # Perform a shallow copy
        attack = copy.copy(self.attack)

        if self.unsqueeze:
            eps = eps.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        attack.eps = eps

        return attack(x, y=y)

    def successful(self, adversarials, y):
        return utils.check_success(self.predict, adversarials, y, False)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)
        
        best_adversarials = x.clone()
        N = x.shape[0]

        eps_lower_bound = torch.ones((N,), device=x.device) * self.min_eps
        eps_upper_bound = torch.ones((N,), device=x.device) * self.max_eps
        last_distances = torch.ones((N), device=x.device) * MAX_DISTANCE

        initial_search_eps = eps_upper_bound.clone()
        for _ in range(self.initial_search_steps):
            adversarials = self.perturb_standard(x, y, initial_search_eps)
            successful = self.successful(adversarials, y)

            distances = utils.adversarial_distance(x, adversarials, self.ord)
            better_distances = distances < last_distances
            replace = successful & better_distances

            best_adversarials[replace] = adversarials[replace]
            last_distances[replace] = distances[replace]

            # Success: Reduce the upper bound
            eps_upper_bound[replace] = initial_search_eps[replace]

            # Halve eps, regardless of the success
            initial_search_eps = initial_search_eps / 2

        for _ in range(self.binary_search_steps):
            eps = (eps_lower_bound + eps_upper_bound) / 2
            adversarials = self.perturb_standard(x, y, eps)
            successful = self.successful(adversarials, y)

            distances = utils.adversarial_distance(x, adversarials, self.ord)
            better_distances = distances < last_distances
            replace = successful & better_distances

            best_adversarials[replace] = adversarials[replace]
            last_distances[replace] = distances[replace]

            # Success: Try again with a lower eps
            eps_upper_bound[replace] = eps[replace]

            # Failure: Try again with a higher eps
            eps_lower_bound[~replace] = eps[~replace]

        return best_adversarials
