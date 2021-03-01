import advertorch

class AdvertorchWrapper(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, inner_attack):
        super().__init__(inner_attack.predict, inner_attack.loss_fn, inner_attack.clip_min, inner_attack.clip_max)
        if hasattr(inner_attack, 'targeted'):
            self.targeted = inner_attack.targeted
        
        self.inner_attack = inner_attack

    def perturb(self, x, **kwargs):
        raise NotImplementedError()