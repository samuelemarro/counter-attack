import advertorch
import torch

import attacks
import utils

# TODO: Remember to always check the .targeted of what you're working with,
# as well as if you're using the standard or defended model

# Nota: In ogni attacco, per "predict" si intende il modello indifeso


class AttackPool(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, evade_detector, attacks, p, brendel_initialization=True):
        self.predict = predict
        self.evade_detector = evade_detector
        assert all(not attack.targeted for attack in attacks)
        assert all((attack.predict == predict) if hasattr(
            attack, 'predict') else True for attack in attacks)
        self.attacks = attacks
        self.p = p
        self.brendel_initialization = brendel_initialization
        self.targeted = False  # Always false

    def successful(self, adversarials, y):
        return utils.misclassified(self.predict, adversarials, y, self.evade_detector)

    def pick_best(self, x, y, pool_results):
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

    def check_brendel(self, attack):
        if isinstance(attack, attacks.BestSampleAttack):
            return self.check_brendel(attack.inner_attack)
        # TODO: Controllare anche RandomTarget e KBest?

        return isinstance(attack, attacks.BrendelBethgeAttack)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        brendel_indices = list([i for i, attack in enumerate(self.attacks) if self.check_brendel(attack)])
        assert len(brendel_indices) <= 1
        brendel_index = None if len(brendel_indices) == 0 else brendel_indices[0]

        if self.brendel_initialization and brendel_index is not None:
            # Use the previous attacks as initialization for the Brendel&Bethge attack

            brendel_attack = self.attacks[brendel_index]
            attacks_without_brendel = [attack for attack in self.attacks if attack is not brendel_attack]
            results_without_brendel = torch.stack([attack.perturb(x, y=y).detach() for attack in attacks_without_brendel], 1)
            best_without_brendel = self.pick_best(x, y, results_without_brendel)

            brendel_results = brendel_attack.perturb(x, y=y, starting_points=best_without_brendel)
            pool_results = torch.stack([torch.stack([no_brendel_result, brendel_result]) for no_brendel_result, brendel_result in zip(best_without_brendel, brendel_results)])

        else:
            pool_results = torch.stack(
                [attack.perturb(x, y=y).detach() for attack in self.attacks], 1)
        
            assert pool_results.shape[1] == len(self.attacks)

        best_adversarials = self.pick_best(x, y, pool_results)
        return best_adversarials
