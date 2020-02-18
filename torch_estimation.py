import torch
import numpy as np

# Nota: Ha bisogno di precisione double, altrimenti gli errori numerici
# causano seri errori nella stima del gradiente (sovrastima del 100000%)

class CoordinateWiseEstimator_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pred_fn, epsilon):
        print(x.requires_grad)
        ctx.save_for_backward(x)
        ctx.pred_fn = pred_fn
        ctx.epsilon = epsilon
        # Necessario per permettere dei forward che usano
        # chiamate a backward()
        with torch.enable_grad():
            return pred_fn(x)

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors[0]

        pred_fn = ctx.pred_fn
        epsilon = ctx.epsilon

        with torch.no_grad():
            estimated_grads = []
            for i in range(x.shape[0]):
                element = x[i]
                output_grad = grad[i]
                estimated_grads.append(CoordinateWiseEstimator_F.estimate_one(element, output_grad, pred_fn, epsilon))

        return torch.stack(estimated_grads), None, None

    @staticmethod
    def _get_noise(shape, dtype):
        N = np.prod(shape).item()
        noise = torch.eye(N, N, dtype=dtype)
        noise = noise.reshape((N,) + shape)
        noise = torch.cat([noise, -noise])
        return noise

    @staticmethod
    def estimate_one(x, output_grad, pred_fn, epsilon):
        noise = CoordinateWiseEstimator_F._get_noise(x.shape, x.dtype).to(x.device)
        N = len(noise)
        print(N)

        theta = x + epsilon * noise
        
        outputs = pred_fn(theta)

        print(output_grad)

        #è corretto?
        print(output_grad.shape)
        outputs = outputs * output_grad

        # TODO: è corretto sommare?
        # Sì, perché il gradiente di una variabile è la somma dei gradienti
        # in base a tutti gli output (che qui denoto f(x) e g(x)), e
        # d/dx f(x) + d/dx g(x) = d/dx (f(x) + g(x))
        loss = outputs.sum(axis=-1)
        
        assert loss.shape == (N,)

        loss = loss.reshape((N,) + (1,) * x.ndim)
        assert loss.ndim == noise.ndim
        gradient = torch.sum(loss * noise, axis=0)
        gradient /= 2 * epsilon

        return gradient

# Versione modulare di CoordinateWiseEstimator_F
class CoordinateWiseEstimator(torch.nn.Module):
    def __init__(self, pred_fn, epsilon=1e-4):
        super().__init__()
        self.pred_fn = pred_fn
        self.epsilon = epsilon

    def forward(self, x):
        return CoordinateWiseEstimator_F.apply(x, self.pred_fn, self.epsilon)

class BinomialSamplingEstimator_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pred_fn, epsilon, count):
        ctx.save_for_backward(x)
        ctx.pred_fn = pred_fn
        ctx.epsilon = epsilon
        ctx.count = count
        return pred_fn(x)

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors[0]

        pred_fn = ctx.pred_fn
        epsilon = ctx.epsilon
        count = ctx.count

        with torch.no_grad():
            estimated_grads = []
            for i in range(x.shape[0]):
                element = x[i]
                output_grad = grad[i]
                estimated_grads.append(BinomialSamplingEstimator_F.estimate_one(element, output_grad, pred_fn, epsilon, count))

        return torch.stack(estimated_grads), None, None, None

    @staticmethod
    def _get_noise(shape, dtype, count):
        sample_shape = (count,) + shape
        noise = torch.distributions.Bernoulli(probs=0.5).sample(sample_shape)
        
        # Discretizzato seguendo https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF
        noise = torch.round(noise)
        noise = 2 * noise - 1
        return noise

    @staticmethod
    def estimate_one(x, output_grad, pred_fn, epsilon, count):
        noise = BinomialSamplingEstimator_F._get_noise(x.shape, x.dtype, count).to(x.device)
        assert noise.shape[0] == count
        assert noise.shape[1:] == x.shape

        positive_theta = x + noise * epsilon
        negative_theta = x - noise * epsilon

        positive_outputs = pred_fn(positive_theta)
        negative_outputs = pred_fn(negative_theta)

        #è corretto?
        print(output_grad)
        positive_outputs = positive_outputs * output_grad
        negative_outputs = negative_outputs * output_grad

        # TODO: è corretto sommare?
        # Sì, perché il gradiente di una variabile è la somma dei gradienti
        # in base a tutti gli output (che qui denoto f(x) e g(x)), e
        # d/dx f(x) + d/dx g(x) = d/dx (f(x) + g(x))
        positive_loss = positive_outputs.sum(axis=-1)
        negative_loss = negative_outputs.sum(axis=-1)

        assert positive_loss.shape == negative_loss.shape

        loss_difference = positive_loss - negative_loss
        print(loss_difference)

        loss_difference = loss_difference.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        print(loss_difference.shape)
        print(noise.shape)
        print((2 * epsilon * noise).shape)

        grad_estimates = loss_difference / (2 * epsilon * noise)


        assert grad_estimates.shape == (count,) + x.shape

        grad = torch.mean(grad_estimates, 0)
        
        assert grad.shape == x.shape

        return grad