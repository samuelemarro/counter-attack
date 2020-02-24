import advertorch
import torch

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
        steps = 50,
        candidates = 10,
        overshoot = 0.02,
        loss = "logits"
    ):
        self.steps = steps
        self.candidates = candidates
        self.overshoot = overshoot
        self.loss = loss

    def _get_loss_fn(
        self, model, classes):

        N = len(classes)
        rows = range(N)
        i0 = classes[:, 0]

        if self.loss == "logits":

            def loss_fun(x, k):
                logits = model(x)
                ik = classes[:, k]
                l0 = logits[rows, i0]
                lk = logits[rows, ik]
                loss = lk - l0
                return loss.sum(), (loss, logits)

        elif self.loss == "crossentropy":

            def loss_fun(x, k):
                logits = model(x)
                ik = classes[:, k]
                l0 = -torch.crossentropy(logits, i0)
                lk = -torch.crossentropy(logits, ik)
                loss = lk - l0
                return loss.sum(), (loss, logits)

        else:
            raise ValueError(
                f"expected loss to be 'logits' or 'crossentropy', got '{self.loss}'"
            )

        return loss_fun

    def perturb(self, x, y=None)::

        criterion = get_criterion(criterion)

        min_, max_ = model.bounds

        logits = model(x)
        classes = logits.argsort(axis=-1).flip(axis=-1)
        if self.candidates is None:
            candidates = logits.shape[-1]  # pragma: no cover
        else:
            candidates = min(self.candidates, logits.shape[-1])
            if not candidates >= 2:
                raise ValueError(  # pragma: no cover
                    f"expected the model output to have atleast 2 classes, got {logits.shape[-1]}"
                )
            logging.info(f"Only testing the top-{candidates} classes")
            classes = classes[:, :candidates]

        N = len(x)
        rows = range(N)

        loss_fun = self._get_loss_fn(model, classes)
        loss_aux_and_grad = torch.value_and_grad_fn(x, loss_fun, has_aux=True)

        x0 = x
        p_total = torch.zeros_like(x)
        for _ in range(self.steps):
            # let's first get the logits using k = 1 to see if we are done
            diffs = [loss_aux_and_grad(x, 1)]
            _, (_, logits), _ = diffs[0]

            is_adv = criterion(x, logits)
            if is_adv.all():
                break

            # then run all the other k's as well
            # we could avoid repeated forward passes and only repeat
            # the backward pass, but this cannot currently be done in eagerpy
            diffs += [loss_aux_and_grad(x, k) for k in range(2, candidates)]

            # we don't need the logits
            diffs_ = [(losses, grad) for _, (losses, _), grad in diffs]
            losses = torch.stack([l for l, _ in diffs_], axis=1)
            grads = torch.stack([g for _, g in diffs_], axis=1)
            assert losses.shape == (N, candidates - 1)
            assert grads.shape == (N, candidates - 1) + x0.shape[1:]

            # calculate the distances
            distances = self.get_distances(losses, grads)
            assert distances.shape == (N, candidates - 1)

            # determine the best directions
            best = distances.argmin(axis=1)
            distances = distances[rows, best]
            losses = losses[rows, best]
            grads = grads[rows, best]
            assert distances.shape == (N,)
            assert losses.shape == (N,)
            assert grads.shape == x0.shape

            # apply perturbation
            distances = distances + 1e-4  # for numerical stability
            p_step = self.get_perturbations(distances, grads)
            assert p_step.shape == x0.shape

            p_total += p_step
            # don't do anything for those that are already adversarial
            x = torch.where(
                atleast_kd(is_adv, x.ndim), x, x0 + (1.0 + self.overshoot) * p_total
            )
            x = torch.clip(x, min_, max_)

        return x

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
        return abs(losses) / (flatten(grads, keep=2).norms.l2(axis=-1) + 1e-8)

    def get_perturbations(self, distances, grads) -> torch.Tensor:
        return (
            atleast_kd(
                distances / (flatten(grads).norms.l2(axis=-1) + 1e-8), grads.ndim,
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
        return abs(losses) / (flatten(grads, keep=2).abs().sum(axis=-1) + 1e-8)

    def get_perturbations(self, distances: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
        return atleast_kd(distances, grads.ndim) * grads.sign()