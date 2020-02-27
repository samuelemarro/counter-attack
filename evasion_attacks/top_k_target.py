import torch

# Importante: predict è il modello non difeso
class TopKTargetEvasionAttack:
    """Chooses the kth top predicted label (by default the 2nd).
    """
    def __init__(self, predict, attack_on_detector_classifier, k=2):
        self.predict = predict

        assert attack_on_detector_classifier.targeted
        self.attack_on_detector_classifier = attack_on_detector_classifier

        assert k > 1
        self.k = k
        self.targeted = False # Always false

    def perturb(self, x, y=None, **kwargs):
        predictions = self.predict(x)
        _, target_labels = torch.topk(predictions, k=self.k)
        target_labels = target_labels[:, -1]

        assert torch.all(torch.logical_not(torch.eq(torch.argmax(predictions, axis=-1), target_labels)))

        return self.attack_on_detector_classifier.perturb(x, y=target_labels, **kwargs)