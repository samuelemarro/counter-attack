import copy

import attacks
import torch


class EpsilonAttack(attacks.AdvertorchWrapper):
    def __init__(self, inner_attack, unsqueeze):
        if not hasattr(inner_attack, 'eps'):
            raise ValueError('inner_attack must have an "eps" attribute.')

        super().__init__(inner_attack)
        self.unsqueeze = unsqueeze

    def perturb(self, x, y=None, eps=None):
        if eps is None:
            return self.inner_attack(x, y=y)

        if isinstance(eps, float):
            eps = torch.ones([len(x)], device=x.device, dtype=x.dtype) * eps

        assert len(eps) == len(x)

        # Perform a shallow copy
        attack = copy.copy(self.inner_attack)

        if self.unsqueeze:
            eps = eps.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        attack.eps = eps

        return attack(x, y=y)