import advertorch
import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.utils import replicate_input, to_one_hot

import utils

"""
This implementation tries to mirror the AdverTorch L2 implementation as closely as
possible, while using the LInf loss and the iterative [..]
The main differences between LInf and L2 are:
- The loss is c * f(x + delta) + sum(max(abs(delta_i) - tau, 0))
- If the attack fails, we multiply c by a constant (default: 2) (L2 multiplies by 10)
- If delta_i < tau for all i, we multiply tau by tau_multiplier, otherwise we terminate
"""

CARLINI_COEFF_UPPER = 1e10
TARGET_MULT = 10000.0
EPS = 1e-6

# TODO: Usa check_success con has_detector=False

class CarliniWagnerLInfAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, num_classes, min_tau=1/256,
                 tau_multiplier=0.9, const_multiplier=2, halve_const=True, confidence=0,
                 targeted=False, learning_rate=0.01,
                 max_iterations=10000,
                 abort_early=True, initial_const=1e-3,
                 clip_min=0., clip_max=1.):
        self.predict = predict
        self.num_classes = num_classes
        self.min_tau = min_tau
        self.tau_multiplier = tau_multiplier
        self.const_multiplier = const_multiplier
        self.halve_const = halve_const
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

    def loss(self, x, delta, y, const, tau):
        adversarials = x + delta
        output = self.predict(adversarials)
        y_onehot = to_one_hot(y, self.num_classes).float()

        real = (y_onehot * output).sum(dim=1)

        other = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)
                 ).max(1)[0]
        # - (y_onehot * TARGET_MULT) is for the true label not to be selected

        #print(real.cpu().detach().numpy())
        #print(other.cpu().detach().numpy())
        if self.targeted:
            loss1 = torch.clamp(other - real + self.confidence, min=0.)
        else:
            loss1 = torch.clamp(real - other + self.confidence, min=0.)

        penalties = torch.clamp(torch.abs(delta) - tau, min=0)

        loss1 = const * loss1#torch.sum(const * loss1)
        loss2 = torch.sum(penalties, dim=(1, 2, 3))#torch.sum(penalties)

        assert loss1.shape == loss2.shape

        loss = loss1 + loss2
        return loss

    def successful(self, adversarials, y):
        return utils.check_success(self.predict, adversarials, y, False)

    # Scales a 0-1 value to clip_min - clip_max range
    def scale_to_bounds(self, value):
        assert (value >= 0).all()
        assert (value <= 1).all()
        return self.clip_min + value * (self.clip_max - self.clip_min)

    def run_attack(self, x, y, initial_const, tau):
        batch_size = len(x)
        best_adversarials = x.clone().detach()
        
        active_samples = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

        ws = torch.nn.Parameter(torch.zeros_like(x))
        optimizer = optim.Adam([ws], lr=self.learning_rate)

        const = initial_const

        while torch.any(active_samples) and const < CARLINI_COEFF_UPPER:
            for i in range(self.max_iterations):
                deltas = self.scale_to_bounds((0.5 + EPS) * (torch.tanh(ws) + 1)) - x
                
                optimizer.zero_grad()
                losses = self.loss(x[active_samples], deltas[active_samples], y[active_samples], const, tau)
                total_loss = torch.sum(losses)

                total_loss.backward()
                optimizer.step()

                adversarials = (x + deltas).detach()
                best_adversarials[active_samples] = adversarials[active_samples]

                # If early aborting is enabled, drop successful samples with small losses
                # (Notice that the current adversarials are saved regardless of whether they are dropped)
                
                if self.abort_early:
                    successful = self.successful(adversarials[active_samples], y[active_samples])
                    small_losses = losses < 0.0001 * const

                    drop = successful & small_losses

                    active_samples[active_samples] = ~drop
                if not active_samples.any():
                    break


            const *= self.const_multiplier
            print('Const: {}'.format(const))

        return best_adversarials


    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        if y is None:
            y = self._get_predicted_label(x)
        
        x = replicate_input(x)
        batch_size = len(x)
        final_adversarials = x.clone()

        active_samples = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

        initial_const = self.initial_const
        tau = 1

        while torch.any(active_samples) and tau >= self.min_tau:
            print('Tau: {}'.format(tau))
            adversarials = self.run_attack(x[active_samples], y[active_samples], initial_const, tau)

            # Drop the failed adversarials (without saving them)
            successful = self.successful(adversarials, y[active_samples])
            active_samples[active_samples] = successful
            final_adversarials[active_samples] = adversarials[successful]

            tau *= self.tau_multiplier
            
            if self.halve_const:
                initial_const /= 2

        return final_adversarials