import logging
from typing import Union, Optional
from typing_extensions import Literal

import advertorch
import foolbox as fb
import numpy as np
import torch

logger = logging.getLogger(__name__)

class FoolboxAttackWrapper(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, model, foolbox_attack, targeted, clip_min=0, clip_max=1):
        # Get the device from the model. This is only possible if the
        # model does not contain both CPU and CUDA tensors
        device = next(model.parameters()).device
        
        self.foolbox_model = fb.models.PyTorchModel(
            model, bounds=(clip_min, clip_max), device=device)
        self.foolbox_attack = foolbox_attack
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)

    def get_criterion(self, y):
        if self.targeted:
            criterion = fb.criteria.TargetedMisclassification(y)
        else:
            criterion = fb.criteria.Misclassification(y)

        return criterion

    def perturb(self, x, y=None, **kwargs):
        x, y = self._verify_and_process_inputs(x, y)

        if 'eps' in kwargs or 'epsilons' in kwargs:
            logger.warning('Passing eps/epsilons to a non-epsilon attack.')

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        criterion = self.get_criterion(y)

        # foolbox_attack returns (adv, clipped_adv, success)
        result = self.foolbox_attack(self.foolbox_model, x, criterion, epsilons=None, **kwargs)
        return result[1]

class BrendelBethgeAttack(FoolboxAttackWrapper):
    def __init__(
            self,
            model,
            p,
            clip_min = 0,
            clip_max = 1,
            targeted = False,
            init_attack = None,
            overshoot: float = 1.1,
            steps: int = 1000,
            lr: float = 1e-3,
            lr_decay: float = 0.5,
            lr_num_decay: int = 20,
            momentum: float = 0.8,
            tensorboard: Union[Literal[False], None, str] = False,
            binary_search_steps: int = 10):
    
        self.init_attack = init_attack

        if p == 2:
            foolbox_attack = fb.attacks.L2BrendelBethgeAttack(
                init_attack=init_attack,
                overshoot=overshoot,
                steps=steps,
                lr=lr,
                lr_decay=lr_decay,
                lr_num_decay=lr_num_decay,
                momentum=momentum,
                tensorboard=tensorboard,
                binary_search_steps=binary_search_steps
            )
        elif np.isinf(p):
            foolbox_attack = fb.attacks.LinfinityBrendelBethgeAttack(
                init_attack=init_attack,
                overshoot=overshoot,
                steps=steps,
                lr=lr,
                lr_decay=lr_decay,
                lr_num_decay=lr_num_decay,
                momentum=momentum,
                tensorboard=tensorboard,
                binary_search_steps=binary_search_steps)
        else:
            raise NotImplementedError('Unsupported metric.')

        super().__init__(model, foolbox_attack, targeted,
                         clip_min=clip_min, clip_max=clip_max)

    def successful(self, x, y):
        same_label = torch.eq(torch.argmax(self.foolbox_model(x), dim=1), y)

        if self.targeted:
            return same_label
        else:
            return ~same_label

    def run_init_attack(self, x, y):
        assert len(x) == len(y)

        if self.init_attack is None:
            # BrendelBethge's default init_attack is LinearSearchBlendedUniformNoiseAttack
            # with default parameters.
            init_attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack()
        else:
            init_attack = self.init_attack
        
        if isinstance(init_attack, fb.attacks.base.Attack):
            criterion = self.get_criterion(y)
            adversarials = init_attack(self.foolbox_model, x, criterion, epsilons=None)[1]
        elif isinstance(init_attack, advertorch.attacks.Attack):
            adversarials = init_attack(x, y=y)
        else:
            raise NotImplementedError
        
        assert adversarials.shape == x.shape

        return adversarials

    def perturb(self, x, y=None, starting_points=None):
        x, y = self._verify_and_process_inputs(x, y)

        if starting_points is None:
            starting_points = self.run_init_attack(x, y)
        else:
            assert starting_points.shape == x.shape

            # Foolbox's implementation of Brendel&Bethge requires all starting points to be successful
            # adversarials. Since that is not always the case, we use Brendel&Bethge's init_attack
            # to initialize the unsuccessful starting_points
            fallback = ~self.successful(starting_points, y)

            fallback_adversarials = self.run_init_attack(x[fallback], y[fallback])
            starting_points[fallback] = fallback_adversarials

        successful_starting = self.successful(starting_points, y)

        num_failures = torch.count_nonzero(~successful_starting)

        if num_failures > 0:
            logger.warning(f'Failed to initialize {num_failures} starting points.')

        adversarials = torch.zeros_like(x)

        # For failed starting points, use the original images (which will be treated as failures)
        adversarials[~successful_starting] = x[~successful_starting]

        # For successful starting points, run the attack and store the results
        computed_adversarials = super().perturb(x[successful_starting], y=y[successful_starting], starting_points=starting_points[successful_starting])
        adversarials[successful_starting] = computed_adversarials

        return adversarials

class DeepFoolAttack(FoolboxAttackWrapper):
    def __init__(
        self,
        model,
        p,
        clip_min=0,
        clip_max=1,
        steps: int = 50,
        candidates: Optional[int] = 10,
        overshoot: float = 0.02,
        loss: Union[Literal["logits"], Literal["crossentropy"]] = "logits",
    ):
        if p == 2:
            foolbox_attack = fb.attacks.L2DeepFoolAttack(
                steps=steps, candidates=candidates, overshoot=overshoot, loss=loss)
        elif np.isinf(p):
            foolbox_attack = fb.attacks.LinfDeepFoolAttack(
                steps=steps, candidates=candidates, overshoot=overshoot, loss=loss)
        else:
            raise NotImplementedError('Unsupported metric.')

        # DeepFool is untargeted
        super().__init__(model, foolbox_attack, False, clip_min=clip_min, clip_max=clip_max)

# This attack is implemented to provide the default Brendel&Bethge behaviour
# in case the initialization from an AttackPool fails.

class LinearSearchBlendedUniformNoiseAttack(FoolboxAttackWrapper):
    def __init__(
        self,
        model,
        clip_min=0,
        clip_max=1,
        targeted=False,
        distance = None,
        directions: int = 1000,
        steps: int = 1000
    ):

        foolbox_attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(
            distance=distance,
            directions=directions,
            steps=steps
        )

        super().__init__(model, foolbox_attack, targeted=targeted, clip_min=clip_min, clip_max=clip_max)