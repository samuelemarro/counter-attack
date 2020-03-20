import advertorch.attacks as attacks

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import batch_l1_proj
from advertorch.utils import replace_active
from advertorch.attacks.utils import rand_init_delta
from advertorch.attacks import Attack
from advertorch.attacks import LabelMixin

import numpy as np
import torch
import torch.nn as nn

import utils

def perform_step(xvar, yvar, outputs, delta, eps, eps_iter, loss_fn, minimize, ord,
                      clip_min, clip_max, l1_sparsity):
    xvar = xvar.detach()
    yvar = yvar.detach()

    loss = loss_fn(outputs, yvar)
    if minimize:
        loss = -loss

    grad = torch.autograd.grad([loss], [delta])[0].detach()
    delta = delta.detach()

    if ord == np.inf:
        grad_sign = grad.sign()
        delta = delta.data + batch_multiply(eps_iter, grad_sign)
        delta = batch_clamp(eps, delta.data)
        delta = clamp(xvar.data + delta.data, clip_min, clip_max
                            ) - xvar

    elif ord == 2:
        grad = normalize_by_pnorm(grad)
        delta = delta.data + batch_multiply(eps_iter, grad)
        delta = clamp(xvar.data + delta.data, clip_min, clip_max
                            ) - xvar.data
        if eps is not None:
            delta = clamp_by_pnorm(delta.data, ord, eps)

    elif ord == 1:
        abs_grad = torch.abs(grad)

        batch_size = grad.size(0)
        view = abs_grad.view(batch_size, -1)
        view_size = view.size(1)
        if l1_sparsity is None:
            vals, idx = view.topk(1)
        else:
            vals, idx = view.topk(
                int(np.round((1 - l1_sparsity) * view_size)))

        out = torch.zeros_like(view).scatter_(1, idx, vals)
        out = out.view_as(grad)
        grad = grad.sign() * (out > 0).float()
        grad = normalize_by_pnorm(grad, p=1)
        delta = delta.data + batch_multiply(eps_iter, grad)

        delta = batch_l1_proj(delta.data.cpu(), eps)
        if xvar.is_cuda:
            delta = delta.data.cuda()
        delta = clamp(xvar.data + delta.data, clip_min, clip_max
                            ) - xvar.data
    else:
        error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
        raise NotImplementedError(error)

    return delta

def er_perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn, targeted,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
                      l1_sparsity=None, early_rejection_threshold=None, return_best=True):
    """
    Iteratively maximize the loss over the input. It is a shared method for
    iterative attacks including IterativeGradientSign, LinfPGD, etc.

    :param xvar: input data.
    :param yvar: input labels.
    :param predict: forward pass function.
    :param nb_iter: number of iterations.
    :param eps: maximum distortion.
    :param eps_iter: attack step size.
    :param loss_fn: loss function.
    :param delta_init: (optional) tensor contains the random initialization.
    :param minimize: (optional bool) whether to minimize or maximize the loss.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param l1_sparsity: sparsity value for L1 projection.
                  - if None, then perform regular L1 projection.
                  - if float value, then perform sparse L1 descent from
                    Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
    :param return_best: if True, return the best adversarials, else return the last
    :return: tensor containing the perturbed input.
    """
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    if isinstance(eps, float):
        eps = torch.ones((len(xvar),), dtype=float, device=xvar.device) * eps
        eps = eps.float()

    active = torch.ones((len(xvar),), dtype=bool, device=xvar.device)

    best_adversarials = xvar.clone()
    best_distances = torch.ones((len(xvar),), device=xvar.device) * np.inf

    for ii in range(nb_iter):
        active_deltas = delta[active].clone()
        active_deltas.requires_grad_()
        active_eps = eps[active]

        adversarials = clamp(xvar[active] + active_deltas, clip_min, clip_max)
        outputs = predict(adversarials)

        if return_best:
            adversarial_labels = torch.argmax(outputs, dim=1)
            if targeted:
                successful = torch.eq(adversarial_labels, yvar[active])
            else:
                successful = ~torch.eq(adversarial_labels, yvar[active])

            distances = utils.adversarial_distance(xvar[active], adversarials, ord)
            better_distance = distances < best_distances[active]
            replace = successful & better_distance

            # Perform replacement only on active samples
            replace_to = active.clone()
            replace_to[active] = replace

            #best_adversarials[replace_to] = adversarials[replace]
            #best_distances[replace_to] = distances[replace]

            replace_active(adversarials, best_adversarials, active, replace)
            replace_active(distances, best_distances, active, replace)

       
        new_delta = perform_step(xvar[active], yvar[active],
                                            outputs, active_deltas,
                                            eps[active], eps_iter, loss_fn,
                                            minimize, ord, clip_min, clip_max, l1_sparsity)

        if early_rejection_threshold is not None:
            # Ignore samples that were early rejected
            # Note: early rejection is computed on the previous adversarials
            # (because filtering active_deltas before the step would mess with gradient propagation)
            rejected = utils.early_rejection(xvar[active], adversarials, yvar[active], outputs, ord, early_rejection_threshold, targeted)
            active[active] = ~rejected
            new_delta = new_delta[~rejected]

            if len(rejected.nonzero()) > 0:
                print(len(rejected.nonzero()))

        if not active.any():
            break

        delta[active] = new_delta

    if return_best:
        return best_adversarials
    else:
        x_adv = clamp(xvar + delta, clip_min, clip_max)
        return x_adv

class ERPGDAttack(Attack, LabelMixin):
    """
    The projected gradient descent attack (Madry et al, 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.
    Paper: https://arxiv.org/pdf/1706.06083.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param ord: (optional) the order of maximum distortion (inf or 2).
    :param targeted: if the attack is targeted.
    :param early_rejection_threshold: the threshold for early rejecting samples
    :param return_best: if True, return the best adversarials, else return the last.
    :param stochastic_consistency: if True, the same image will always use the same random
        values.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, l1_sparsity=None, targeted=False,
            early_rejection_threshold=None, return_best=True,
            stochastic_consistency=False):
        """
        Create an instance of the ERPGDAttack.

        """
        super(ERPGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.l1_sparsity = l1_sparsity

        self.early_rejection_threshold = early_rejection_threshold
        self.return_best = return_best
        self.stochastic_consistency = stochastic_consistency

        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        if self.rand_init:
            if self.stochastic_consistency:
                utils.consistent_rand_init_delta(delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            else:
                rand_init_delta(
                    delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = er_perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, targeted=self.targeted,
            minimize=self.targeted, ord=self.ord,
            clip_min=self.clip_min, clip_max=self.clip_max,
            delta_init=delta, l1_sparsity=self.l1_sparsity,
            early_rejection_threshold=self.early_rejection_threshold,
            return_best=self.return_best
        )

        return rval.data


class LinfERPGDAttack(ERPGDAttack):
    """
    PGD Attack with order=Linf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param early_rejection_threshold: the threshold for early rejecting samples
    :param return_best: if True, return the best adversarials, else return the last.
    :param stochastic_consistency: if True, the same image will always use the same random
        values.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, early_rejection_threshold=None, return_best=True,
            stochastic_consistency=False):
        ord = np.inf
        super(LinfERPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted, ord=ord,
            early_rejection_threshold=early_rejection_threshold,
            return_best=return_best, stochastic_consistency=stochastic_consistency)


class L2ERPGDAttack(ERPGDAttack):
    """
    PGD Attack with order=L2

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param rand_init: (optional bool) random initialization.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param early_rejection_threshold: the threshold for early rejecting samples.
    :param return_best: if True, return the best adversarials, else return the last.
    :param stochastic_consistency: if True, the same image will always use the same random
        values.
    """

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, early_rejection_threshold=None, return_best=True,
            stochastic_consistency=False):
        ord = 2
        super(L2ERPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, targeted=targeted,
            ord=ord, early_rejection_threshold=early_rejection_threshold,
            return_best=return_best, stochastic_consistency=stochastic_consistency)

class L2ERBasicIterativeAttack(ERPGDAttack):
    """Like GradientAttack but with several steps for each epsilon.

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param early_rejection_threshold: the threshold for early rejecting samples
    :param return_best: if True, return the best adversarials, else return the last
    """

    def __init__(self, predict, loss_fn=None, eps=0.1, nb_iter=10,
                 eps_iter=0.05, clip_min=0., clip_max=1., targeted=False,
                 early_rejection_threshold=None, return_best=True):
        ord = 2
        rand_init = False
        l1_sparsity = None
        stochastic_consistency = False
        super(L2ERBasicIterativeAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, l1_sparsity, targeted, early_rejection_threshold,
            return_best, stochastic_consistency)


class LinfERBasicIterativeAttack(ERPGDAttack):
    """
    Like GradientSignAttack but with several steps for each epsilon.
    Aka Basic Iterative Attack.
    Paper: https://arxiv.org/pdf/1611.01236.pdf

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: maximum distortion.
    :param nb_iter: number of iterations.
    :param eps_iter: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: if the attack is targeted.
    :param early_rejection_threshold: the threshold for early rejecting samples
    :param return_best: if True, return the best adversarials, else return the last
    """

    def __init__(self, predict, loss_fn=None, eps=0.1, nb_iter=10,
                 eps_iter=0.05, clip_min=0., clip_max=1., targeted=False,
                 early_rejection_threshold=None, return_best=True):
        ord = np.inf
        rand_init = False
        l1_sparsity = None
        stochastic_consistency = False
        super(LinfERBasicIterativeAttack, self).__init__(
            predict, loss_fn, eps, nb_iter, eps_iter, rand_init,
            clip_min, clip_max, ord, l1_sparsity, targeted,
            early_rejection_threshold, return_best, stochastic_consistency)