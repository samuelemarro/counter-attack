import torch

import utils

# TODO: Controllare

# Importante: undefended_model è il modello non difeso
class RandomTargetEvasionAttack:
    """Chooses a random label that is not the
    original one.
    """
    def __init__(self, undefended_model, attack_on_detector_classifier, stochastic_consistency=False):
        self.undefended_model = undefended_model
        
        assert attack_on_detector_classifier.targeted
        self.predict = attack_on_detector_classifier.predict
        self.attack_on_detector_classifier = attack_on_detector_classifier
        self.stochastic_consistency = stochastic_consistency
        self.targeted = False # Always false

    def perturb(self, x, y=None, **kwargs):
        predictions = self.undefended_model(x)

        assert len(predictions) == len(x)

        num_classes = predictions.shape[1]

        original_labels = torch.argmax(predictions, axis=-1)

        target_labels = []

        for i, original_label in enumerate(original_labels):
            target_label = None

            while target_label is None or target_label == original_label:
                if self.stochastic_consistency:
                    # TODO: Controllare consistenza
                    target_label = utils.consistent_randint(x[i], 0, num_classes, (1,), x.device)
                else:
                    target_label = torch.randint(0, num_classes, (1,))

            target_labels.append(target_label)

        target_labels = torch.cat(target_labels).to(original_labels.device)

        assert torch.all(torch.logical_not(torch.eq(original_labels, target_labels)))

        return self.attack_on_detector_classifier.perturb(x, y=target_labels, **kwargs).detach()