import logging

import advertorch
import torch

import utils

logger = logging.getLogger(__name__)

# Nota: Usa un boolean indexing, ma rimuoverlo richiederebbe riscriverlo in maniera poco
# elegante. Se viene usato in codice GPU critico, riscriverlo.

class AttackPool(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, evade_detector, pool_attacks, p, targeted=False, clip_min=0, clip_max=1):
        super().__init__(predict, None, clip_min=clip_min, clip_max=clip_max)
        self.evade_detector = evade_detector

        assert all((attack.targeted == targeted) for attack in pool_attacks)
        assert all((attack.predict == predict) if hasattr(
            attack, 'predict') else True for attack in pool_attacks)
        assert all((attack.clip_min == clip_min) if hasattr(
            attack, 'clip_min') else True for attack in pool_attacks)
        assert all((attack.clip_max == clip_max) if hasattr(
            attack, 'clip_max') else True for attack in pool_attacks)
        assert len(pool_attacks) > 0

        if len(pool_attacks) == 1:
            logger.warning('Creating an AttackPool with only one attack.')

        logger.debug('Creating attack pool with %s attacks.', len(pool_attacks))

        self.pool_attacks = pool_attacks
        self.p = p
        self.targeted = targeted

    def successful(self, pool_results, label):
        adversarial_outputs = self.predict(pool_results).detach()
        adversarial_labels = torch.argmax(adversarial_outputs, dim=1)

        if self.targeted:
            return torch.eq(adversarial_labels, label)
        else:
            return ~torch.eq(adversarial_labels, label)

    def pick_best(self, x, y, pool_results):
        assert len(pool_results) == len(x)

        best_adversarials = []

        for original, pool_result, label in zip(x, pool_results, y):
            successful = self.successful(pool_result, label).detach()
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
                # All the pool_attacks failed: Return the original
                logger.debug('All the pool attacks failed.')
                best_adversarials.append(original)

        best_adversarials = torch.stack(best_adversarials)

        assert best_adversarials.shape == x.shape

        return best_adversarials
    
    def perturb(self, x, y=None, **kwargs):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        pool_results = torch.stack(
            [attack.perturb(x, y=y, **kwargs).detach() for attack in self.pool_attacks], 1)

        assert len(pool_results) == len(x)
        assert pool_results.shape[1] == len(self.pool_attacks)

        best_adversarials = self.pick_best(x, y, pool_results)
        return best_adversarials
