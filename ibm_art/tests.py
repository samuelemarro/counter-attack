import test_utils
import counter_attack

import art

class AttackConfiguration:
    def __init__(self, name, generator):
        self.name = name
        self.generator = generator

class AttackResult:
    def __init__(self, attack_configuration, images, adversarials, original_count, correct_count, p):
        self.attack_configuration = attack_configuration
        self.images = images
        self.adversarials = adversarials
        self.original_count = original_count
        self.correct_count = correct_count
        self.p = p

class EvasionResult:
    def __init__(self, inner_attack_configuration, outer_attack_configuration, images, adversarials, original_count, correct_count, p, counter_attack_threshold):
        self.inner_attack_configuration = inner_attack_configuration
        self.outer_attack_configuration = outer_attack_configuration
        self.images = images
        self.adversarials = adversarials
        self.original_count = original_count
        self.correct_count = correct_count
        self.p = p
        self.counter_attack_threshold = counter_attack_threshold

    def distances(self):
        pass

def counter_attack_matrix(classifier, images, labels, inner_attack_configurations, outer_attack_configurations, p, threshold, detector_estimator):
    evasion_results = []
    original_count = len(images)
    images, labels = test_utils.remove_misclassified(classifier, images, labels)
    correct_count = len(images)

    for inner_attack_configuration in inner_attack_configurations:
        inner_attack = inner_attack_configuration.generator(classifier)

        detector = counter_attack.CounterAttackDetector(classifier, inner_attack, p, threshold, detector_estimator)
        detector_classifier = art.classifiers.DetectorClassifier(classifier, detector)

        for outer_attack_configuration in outer_attack_configurations:
            outer_attack = outer_attack_configuration.generator(detector_classifier)

            adversarials = outer_attack.generate(images)

            evasion_result = EvasionResult(inner_attack_configuration, outer_attack_configuration, images, adversarials, original_count, correct_count, p, counter_attack_threshold)

            evasion_results.append(evasion_result)


    return evasion_results

# Qual è la distribuzione degli score su immagini che ingannano il detector?
def transfer_test(target_classifier, detector):
    pass
