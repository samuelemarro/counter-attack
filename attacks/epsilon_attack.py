import copy

import attacks
import torch


class EpsilonAttack(attacks.AdvertorchWrapper):
    def __init__(self, inner_attack, unsqueeze, force_eps=True):
        if not hasattr(inner_attack, 'eps'):
            raise ValueError('inner_attack must have an "eps" attribute.')

        super().__init__(inner_attack)
        self.unsqueeze = unsqueeze
        self.force_eps = force_eps

    def perturb(self, x, y=None, eps=None):
        if eps is None:
            if self.force_eps:
                raise RuntimeError('Received a perturb() call without eps. If this is intentional, '
                'initialize EpsilonAttack with force_eps=False.')

            return self.inner_attack(x, y=y)

        if isinstance(eps, float):
            eps = torch.ones([len(x)], device=x.device, dtype=x.dtype) * eps

        assert len(eps) == len(x)

        # Perform a shallow copy
        attack = copy.copy(self.inner_attack)

        if self.unsqueeze:
            assert len(eps.shape) == 1
            eps = eps.reshape([len(eps)] + ([1] * (len(x.shape) - 1)))

        attack.eps = eps

        return attack(x, y=y)
