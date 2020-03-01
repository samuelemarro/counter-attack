import advertorch
import torch
from advertorch.utils import replicate_input, to_one_hot

import utils

def atleast_kd(x, k):
    shape = x.shape + (1,) * (k - x.ndim)
    return x.reshape(shape)

# TODO: Controllare che l'implementazione sia corretta
# TODO: Sistemare supporto detector
# TODO: è normale che riceva direttamente y= ?


# Per l'implementazione:
# Success deve controllare che la label non sia rejected
# Credo si possa fare che ignori completamente la label rejected
# In alternativa, si può fitrare da classes, oppure si può scegliere un best diverso

class DeepFoolAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    """A simple and fast gradient-based adversarial attack.

    Implements the `DeepFool`_ attack.

    Args:
        steps : Maximum number of steps to perform.
        candidates : Limit on the number of the most likely classes that should
            be considered. A small value is usually sufficient and much faster.
        overshoot : How much to overshoot the boundary.
        loss  Loss function to use inside the update function.


    .. _DeepFool:
            Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
            "DeepFool: a simple and accurate method to fool Deep neural
            networks", https://arxiv.org/abs/1511.04599

    """

    def __init__(
        self,
        predict,
        evade_detector,
        steps = 50,
        candidates = None,
        overshoot = 0.02,
        clip_min = 0,
        clip_max = 1
    ):
        self.predict = predict
        self.evade_detector = evade_detector
        self.steps = steps
        self.candidates = candidates
        self.overshoot = overshoot

        self.clip_min = 0
        self.clip_max = 1

        self.targeted = False # Always false

    def loss(self, x, classes, k):
        N = len(classes)
        rows = range(N)
        i0 = classes[:, 0]

        logits = self.predict(x)
        ik = classes[:, k]
        l0 = logits[rows, i0]
        lk = logits[rows, ik]
        loss = lk - l0
        return loss

    def successful(self, adversarials, y):
        return utils.check_success(self.predict, adversarials, y, self.evade_detector)

    # TODO: Che succede se la classe più alta di partenza è rejected?
    # Che succede se si passa da originale a rejected?
    # La tecnica di proiezione che ho scelto ha senso?
    # DeepFool se si trova nella zona rejected può scegliere di proiettare verso
    # la label originale! (Dunque best non può essere la label originale)

    def perturb(self, x, y=None):
        x = replicate_input(x)

        logits = self.predict(x)
        classes = logits.argsort(axis=-1).flip(-1)

        if self.candidates is None:
            candidates = logits.shape[-1]  # pragma: no cover
        else:
            candidates = min(self.candidates, logits.shape[-1])
            if not candidates >= 2:
                raise ValueError(  # pragma: no cover
                    f"expected the model output to have atleast 2 classes, got {logits.shape[-1]}"
                )
            classes = classes[:, :candidates]

        N = len(x)
        rows = range(N)

        adversarials = x.clone().detach()

        p_total = torch.zeros_like(x)
        for _ in range(self.steps):
            is_adv = self.successful(adversarials, y)
            if is_adv.all():
                break

            adv_copy = adversarials.detach().clone()
            adv_copy.requires_grad = True

            losses = [self.loss(adv_copy, classes, k) for k in range(1, candidates)]

            grads = torch.stack([torch.autograd.grad(loss, adv_copy, torch.ones_like(loss))[0] for loss in losses], axis=1)
            losses = torch.stack(losses, axis=1)
            
            assert losses.shape == (N, candidates - 1)
            assert grads.shape == (N, candidates - 1) + x.shape[1:]

            # Calculate the distances
            distances = self.get_distances(losses, grads)
            assert distances.shape == (N, candidates - 1)

            # Determine the best directions
            if self.evade_detector:
                rejected_label = logits.shape[1] - 1
                sorted_labels = torch.argsort(dim=1)
                
                best = sorted_labels[:, 0]

                # If the best label is "rejected" or is the original label, replace it with the 2nd-best
                # TODO: What if best and 2nd-best are rejected/original?
                rejected = torch.eq(best, rejected_label)
                original = torch.eq(best, y)
                replace = rejected | original
                assert replace.shape == (len(best),)

                second_best = sorted_labels[:, 1]
                best[replace] = second_best[replace]
            else:
                best = distances.argmin(axis=1)

            # TODO: rows -> : ?
            distances = distances[rows, best]
            losses = losses[rows, best]
            grads = grads[rows, best]
            assert distances.shape == (N,)
            assert losses.shape == (N,)
            assert grads.shape == x.shape

            # Apply perturbation
            distances = distances + 1e-4  # For numerical stability
            p_step = self.get_perturbations(distances, grads)
            assert p_step.shape == x.shape

            p_total += p_step

            # Don't do anything for those that are already adversarial
            adversarials = torch.where(
                atleast_kd(is_adv, adversarials.ndim), adversarials, x + (1.0 + self.overshoot) * p_total
            )

            adversarials = torch.clamp(adversarials, self.clip_min, self.clip_max)

        return adversarials

    def get_distances(self, losses, grads):
        raise NotImplementedError

    def get_perturbations(self, distances, grads):
        raise NotImplementedError


class L2DeepFoolAttack(DeepFoolAttack):
    """A simple and fast gradient-based adversarial attack.

    Implements the DeepFool L2 attack. [#Moos15]_

    Args:
        steps : Maximum number of steps to perform.
        candidates : Limit on the number of the most likely classes that should
            be considered. A small value is usually sufficient and much faster.
        overshoot : How much to overshoot the boundary.
        loss  Loss function to use inside the update function.

    References:
        .. [#Moos15]: Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
            "DeepFool: a simple and accurate method to fool Deep neural
            networks", https://arxiv.org/abs/1511.04599

    """

    def get_distances(self, losses, grads):
        return abs(losses) / (grads.flatten(start_dim=2).norm(p=2, dim=-1) + 1e-8)

    def get_perturbations(self, distances, grads):
        return (
            atleast_kd(
                distances / (grads.flatten().norm(p=2, dim=-1) + 1e-8), grads.ndim,
            )
            * grads
        )


class LInfDeepFoolAttack(DeepFoolAttack):
    """A simple and fast gradient-based adversarial attack.

        Implements the `DeepFool`_ L-Infinity attack.

        Args:
            steps : Maximum number of steps to perform.
            candidates : Limit on the number of the most likely classes that should
                be considered. A small value is usually sufficient and much faster.
            overshoot : How much to overshoot the boundary.
            loss  Loss function to use inside the update function.


        .. _DeepFool:
                Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard,
                "DeepFool: a simple and accurate method to fool Deep neural
                networks", https://arxiv.org/abs/1511.04599

        """

    def get_distances(self, losses: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
        return abs(losses) / (grads.flatten(start_dim=2).abs().sum(axis=-1) + 1e-8)

    def get_perturbations(self, distances: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
        return atleast_kd(distances, grads.ndim) * grads.sign()