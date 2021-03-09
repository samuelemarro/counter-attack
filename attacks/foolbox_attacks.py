import logging
from typing import Union, Optional
from typing_extensions import Literal

import advertorch
import foolbox as fb
import numpy as np
import torch

logger = logging.getLogger(__name__)

class FoolboxAttackWrapper(advertorch.attacks.LabelMixin):
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
            logger.warning('You are passing eps/epsilons to a non-epsilon attack. Is this intentional?')

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)

        criterion = self.get_criterion(y)

        # foolbox_attack returns (adv, clipped_adv, success)
        return self.foolbox_attack(self.foolbox_model, x, criterion, epsilons=None, **kwargs)[1]

class BrendelBethgeAttack(FoolboxAttackWrapper):
    def __init__(
            self,
            model,
            p,
            clip_min=0,
            clip_max=1,
            targeted=False,
            init_attack: Optional[fb.attacks.base.MinimizationAttack] = None,
            overshoot: float = 1.1,
            steps: int = 1000,
            lr: float = 1e-3,
            lr_decay: float = 0.5,
            lr_num_decay: int = 20,
            momentum: float = 0.8,
            tensorboard: Union[Literal[False], None, str] = False,
            binary_search_steps: int = 10):
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

        self.torch_model = model

        super().__init__(model, foolbox_attack, targeted,
                         clip_min=clip_min, clip_max=clip_max)

    def perturb(self, x, y=None, starting_points=None):
        x, y = self._verify_and_process_inputs(x, y)
        # Foolbox's implementation of Brendel&Bethge requires all starting points to be successful
        # adversarials. Since that is not always the case, we use Brendel&Bethge's init_attack
        # to initialize the unsuccessful starting_points
        if starting_points is not None:
            same_label = torch.eq(torch.argmax(self.foolbox_model(starting_points), dim=1), y)

            # fallback is equal to ~successful
            if self.targeted:
                fallback = ~same_label
            else:
                fallback = same_label

            if fallback.any():
                criterion = self.get_criterion(y)

                if self.foolbox_attack.init_attack is None:
                    # BrendelBethge's default init_attack is LinearSearchBlendedUniformNoiseAttack
                    # with default parameters.
                    init_attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack()
                else:
                    init_attack = self.foolbox_attack.init_attack

                fallback_adversarials = init_attack(self.foolbox_model, x, criterion, epsilons=None)[1]

                starting_points[fallback] = fallback_adversarials[fallback]

        return super().perturb(x, y=y, starting_points=starting_points)


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

class CarliniWagnerL2Attack(FoolboxAttackWrapper):
    def __init__(
        self,
        model,
        clip_min=0,
        clip_max=1,
        targeted=False,
        binary_search_steps: int = 9,
        steps: int = 10000,
        stepsize: float = 1e-2,
        confidence: float = 0,
        initial_const: float = 1e-3,
        abort_early: bool = True,
    ):
        foolbox_attack = fb.attacks.L2CarliniWagnerAttack(
            binary_search_steps=binary_search_steps,
            steps=steps,
            stepsize=stepsize,
            confidence=confidence,
            initial_const=initial_const,
            abort_early=abort_early)

        super().__init__(model, foolbox_attack, targeted,
                         clip_min=clip_min, clip_max=clip_max)


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