import advertorch
import torch
import torch.nn as nn

from advertorch.utils import is_float_or_torch_tensor
import advertorch.attacks.iterative_projected_gradient as ipg

import numpy as np

class RandomTargetEvasionAttack:
    """Chooses a random label that is not the
    original one.
    """
    def __init__(self, classifier, attack_on_detector_classifier):
        self.classifier = classifier
        self.attack_on_detector_classifier = attack_on_detector_classifier

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
        self.attack_on_detector_classifier = attack_on_detector_classifier
        self.k = k

    def perturb(self, x, y=None, **kwargs):
        predictions = self.classifier(x)
        _, target_labels = torch.topk(predictions, k=self.k)
        target_labels = target_labels[:, -1]

        assert torch.all(torch.logical_not(torch.eq(torch.argmax(predictions, axis=-1), target_labels)))

        return self.attack_on_detector_classifier.perturb(x, y=target_labels, **kwargs)

class AttackPool:
    def __init__(self, attacks, p):
        self.attacks = attacks
        self.p = p

    def perturb(self, x, y=None):
        pool_results = torch.stack([attack.perturb(x, y=y) for attack in self.attacks], 1)

        assert pool_results.shape[1] == len(self.attacks)
        assert len(pool_results) == len(x)
        
        best_adversarials = []

        for original, pool_result in zip(x, pool_results):
            # Bisogna prima appiattire affinché pairwise_distance calcoli correttamente
            # (original è singolo, pool_result è una batch)
            distances = torch.pairwise_distance(original.flatten(), pool_result.flatten(1), self.p)

            assert distances.shape == (len(pool_result),)

            best_distance_index = torch.argmin(distances)
            
            best_adversarials.append(pool_result[best_distance_index])

        best_adversarials = torch.stack(best_adversarials)

        assert best_adversarials.shape == x.shape

        return best_adversarials


class PGDBinarySearch(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(
            self, predict, ord, min_eps=0, max_eps=1, binary_search_steps=9,
            loss_fn=None, nb_iter=40, eps_iter=0.01, rand_init=True,
            clip_min=0., clip_max=1., l1_sparsity=None,
            targeted=False):

        super().__init__(
            predict, loss_fn, clip_min, clip_max)

        self.min_eps = min_eps
        self.max_eps = max_eps
        self.binary_search_steps = binary_search_steps
        
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity
        assert is_float_or_torch_tensor(self.eps_iter)

    def successful(self, adversarials, y):
        predicted_labels = torch.argmax(self.predict(adversarials), axis=1)

        assert predicted_labels.shape == y.shape

        return ~torch.eq(predicted_labels, y)

    def perturb_standard(self, x, y, eps):
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            ipg.rand_init_delta(
                delta, x, self.ord, eps, self.clip_min, self.clip_max)
            delta.data = torch.clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = ipg.perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta,
            l1_sparsity=self.l1_sparsity,
        )

        return rval.data

    def perturb(self, x, y=None):
        best_adversarials = x.clone()
        N = x.shape[0]

        eps_lower_bound = torch.ones((N,), device=x.device) * self.min_eps
        eps_upper_bound = torch.ones((N,), device=x.device) * self.max_eps

        for _ in range(self.binary_search_steps):
            eps = (eps_lower_bound + eps_upper_bound) / 2
            adversarials = self.perturb_standard(x, y, eps)
            successful = self.successful(adversarials, y)

            # TODO: Controllare effettivamente se sono migliori?
            best_adversarials[successful] = adversarials[successful]

            # Success: Try again with a lower eps
            eps_upper_bound[successful] = eps[successful]

            # Failure: Try again with a higher eps
            eps_lower_bound[~successful] = eps[~successful]

        return best_adversarials
