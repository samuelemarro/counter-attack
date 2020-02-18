import art
import art.attacks as attacks

import numpy as np

class RandomTargetAttackAgainstDetector(attacks.EvasionAttack):
    def __init__(self, classifier, attack_on_detector_classifier):
        super().__init__(classifier)
        self.attack_on_detector_classifier = attack_on_detector_classifier

    def generate(self, x, y=None, **kwargs):
        assert y is None # Rimuovere?

        original_labels = np.argmax(self.classifier.predict(x), axis=-1)

        target_labels = art.utils.random_targets(original_labels, self.classifier.nb_classes())

        return self.attack_on_detector_classifier.generate(x, y=target_labels, **kwargs)


class SecondBestTargetAttackAgainstDetector(attacks.EvasionAttack):
    def __init__(self, classifier, attack_on_detector_classifier):
        super().__init__(classifier)
        self.attack_on_detector_classifier = attack_on_detector_classifier

    def generate(self, x, y=None, **kwargs):

        original_labels = np.argmax(self.classifier.predict(x), axis=-1)

        target_labels = art.utils.second_most_likely_class(original_labels, self.classifier.nb_classes())

        return self.attack_on_detector_classifier.generate(x, y=target_labels, **kwargs)

#class CarliniL2DetectorLoss(attacks.CarliniL2Method)
