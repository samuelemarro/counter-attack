import torch

# TODO: Prendere num_classes in input?
# TODO: Controllare

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