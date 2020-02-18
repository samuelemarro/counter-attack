import advertorch
import torch

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
