import advertorch
import torch

import utils

# TODO: Remember to always check the .targeted of what you're working with,
# as well as if you're using the standard or defended model

# Nota: In ogni attacco, per "predict" si intende il modello indifeso


class AttackPool(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, evade_detector, attacks, p):
        self.predict = predict
        self.evade_detector = evade_detector
        assert all(not attack.targeted for attack in attacks)
        assert all((attack.predict == predict) if hasattr(
            attack, 'predict') else True for attack in attacks)
        self.attacks = attacks
        self.p = p
        self.targeted = False  # Always false

    def successful(self, adversarials, y):
        return utils.misclassified(self.predict, adversarials, y, self.evade_detector)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        pool_results = torch.stack(
            [attack.perturb(x, y=y).detach() for attack in self.attacks], 1)

        assert pool_results.shape[1] == len(self.attacks)
        assert len(pool_results) == len(x)

        best_adversarials = []

        for original, pool_result, label in zip(x, pool_results, y):
            expanded_label = label.expand(len(pool_result))

            successful = self.successful(pool_result, expanded_label)
            assert successful.shape == (len(successful),)

            if successful.any():
                successful_adversarials = pool_result[successful]

                distances = utils.one_many_adversarial_distance(
                    original, successful_adversarials, self.p)

                assert distances.shape == (len(successful_adversarials),)

                best_distance_index = torch.argmin(distances)

                best_adversarials.append(
                    successful_adversarials[best_distance_index])
            else:
                # All the attacks failed: Return the original
                best_adversarials.append(original)

        best_adversarials = torch.stack(best_adversarials)

        assert best_adversarials.shape == x.shape

        return best_adversarials
