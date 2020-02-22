import advertorch
import torch
import torch.nn as nn
import torch.optim as optim

from advertorch.utils import replicate_input, to_one_hot


"""
This implementation tries to mirror the AdverTorch L2 implementation as closely as
possible, while using the LInf loss and the iterative [..]
The main differences between LInf and L2 are:
- The loss is c * f(x + delta) + sum(max(abs(delta_i) - tau, 0))
- If the attack fails, we multiply c by a constant (default: 2) (L2 multiplies by 10)
- If delta_i < tau for all i, we multiply tau by tau_multiplier, otherwise we terminate
"""

# L'attacco è nel TanH space
# Usi arctanh per andare nel Tanh space, tanh per tornare nello spazio normale
#anche abs(delta_i) si riferisce a delta_i nello spazio TanH

CARLINI_L2DIST_UPPER = 1e10
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL = -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10
EPS = 1e-6

# IMPORTANTE: min_tau è la minima distanza restituita da CW
# Questo significa che la distanza minima di default sarà 1/256 (cioè 3.9 * 10^-3)


# Const viene modificata in due momenti:
# Nel loop "inner" (in cui si cerca la più piccola const che fa aver successo),
# si parte da una const iniziale che se fallisce viene moltiplicata per
# const_factor (solitamente 2)
# Nel loop "outer" si passa al loop inner la const. La const iniziale è 1e-5, e se
# reduce_const è attivo ogni volta che il loop inner ha successo la const iniziale viene dimezzata

# Il loop outer si ferma non appena il loop inner fallisce

# L'inner loop usa moltiplicazione standard per trovare const, non binary search

# TODO: Non sto usando il clipping (ha utilizzi nel rescale)

class CarliniWagnerLInfAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, num_classes, min_tau=1/256,
                 tau_multiplier=0.9, const_multiplier=2, confidence=0,
                 targeted=False, learning_rate=0.01,
                 max_iterations=10000,
                 abort_early=True, initial_const=1e-3,
                 clip_min=0., clip_max=1.):
        self.predict = predict
        self.num_classes = num_classes
        self.min_tau = min_tau
        self.tau_multiplier = tau_multiplier
        self.const_multiplier = const_multiplier
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        # TODO: Togliere
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

        if self.targeted:
            loss1 = torch.clamp(other - real + self.confidence, min=0.)
        else:
            loss1 = torch.clamp(real - other + self.confidence, min=0.)

        penalties = torch.clamp(delta - tau, min=0)

        loss1 = const * loss1#torch.sum(const * loss1)
        loss2 = torch.sum(penalties, dim=(1, 2, 3))#torch.sum(penalties)

        loss = loss1 + loss2
        return loss

    def successful(self, adversarials, y):
        predicted_labels = torch.argmax(self.predict(adversarials), axis=1)

        assert predicted_labels.shape == y.shape

        return ~torch.eq(predicted_labels, y)
    def run_attack(self, x, y, initial_const, tau):
        batch_size = len(x)
        best_adversarials = x
        active_samples = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

        ws = torch.nn.Parameter(torch.zeros_like(x))
        optimizer = optim.Adam([ws], lr=self.learning_rate)

        const = initial_const

        while torch.any(active_samples) and const < CARLINI_COEFF_UPPER:
            for i in range(self.max_iterations):
                deltas = (0.5 + EPS) * (torch.tanh(ws) + 1) - x
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
                    #print(drop)
                    #print(losses)


                    active_samples[active_samples] = ~drop
                if not active_samples.any():
                    break
                #print(len(torch.nonzero(active_samples)))


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
        final_advs = x

        active_samples = torch.ones((batch_size,), dtype=torch.bool, device=x.device)

        initial_const = self.initial_const
        tau = 1

        while torch.any(active_samples) and tau >= self.min_tau:
            print('Tau: {}'.format(tau))
            adversarials = self.run_attack(x[active_samples], y[active_samples], initial_const, tau)

            # TODO: CONTROLLARE
            # Drop the failed adversarials (without saving them)
            successful = self.successful(adversarials, y[active_samples])
            drop = torch.logical_not(successful)

            active_samples[active_samples] = ~drop
            final_advs[active_samples] = adversarials[successful]


            tau *= self.tau_multiplier
            # TODO: Renderlo opzionale?
            initial_const /= 2


        return final_advs

    

    
# Nuovo parametro: Tau
# Tau deve decadere con le iterazioni

# IBM Art usa tanh, anche se non è nel paper
# IBM Art ha anche eps, un upperbound sulla distanza L0 (è 0.3, quindi suppongo sia la percentuale di pixel)

# Implementazione originale: https://github.com/carlini/nn_robust_attacks/blob/master/li_attack.py

    