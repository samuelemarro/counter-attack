import numpy as np

#http://www.jmlr.org/papers/volume18/15-592/15-592.pdf

# Calcola il gradiente dello score rispetto agli input
# Le tecniche standard (incluso substitute.gradient()) calcolano
# il gradiente della Cross-Entropy Loss
class DetectorEstimator:
    def gradient(self, inputs, pred_fn, bounds):
        gradients = []
        for x in inputs:
            print(x.shape)
            gradients.append(self.gradient_one(x, pred_fn, bounds))
        gradients = np.array(gradients)

        return gradients

    def gradient_one(self, x, pred_fn, bounds):
        raise NotImplementedError()

# Variante di CoordinateWiseGradientEstimator che calcola la derivata
# dello score
class CoordinateWiseDetectorEstimator(DetectorEstimator):
    def __init__(self, epsilon, clip=True):
        self.epsilon = epsilon
        self.clip = clip

    def _get_noise(self, shape, dtype):
        N = np.prod(shape)
        noise = np.eye(N, N, dtype=dtype)
        noise = noise.reshape((N,) + shape)
        noise = np.concatenate([noise, -noise])
        return noise

    def gradient_one(self, x, pred_fn, bounds):
        noise = self._get_noise(x.shape, x.dtype)
        N = len(noise)

        min_, max_ = bounds
        scaled_epsilon = self.epsilon * (max_ - min_)

        theta = x + scaled_epsilon * noise
        if self.clip:
            theta = np.clip(theta, min_, max_)
        loss = pred_fn(theta)
        assert loss.shape == (N,)

        loss = loss.reshape((N,) + (1,) * x.ndim)
        assert loss.ndim == noise.ndim
        gradient = np.sum(loss * noise, axis=0)
        gradient /= 2 * scaled_epsilon

        return gradient

# Ispirato a Simultaneous perturbation stochastic approximation
# https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation

# Problema: Talvolta restituisce gradienti infiniti (perché divide per zero)

class RandomSamplingDetectorEstimator(DetectorEstimator):
    def __init__(self, epsilon, clip=True):
        self.epsilon = epsilon
        self.clip = clip

    def _get_noise(self, shape, dtype):
        noise = np.random.binomial(1, 0.5, size=shape).astype(np.float32)
        bigger_than_pointfive = noise >= 0.5
        noise[bigger_than_pointfive] = 1
        noise[np.logical_not(bigger_than_pointfive)] = -1

        noise = np.stack([noise, -noise], axis=0)

        return noise

    def gradient_one(self, x, pred_fn, bounds):
        gradient_estimates = []
        for _ in range(200):
            noise = self._get_noise(x.shape, x.dtype)

            min_, max_ = bounds
            scaled_epsilon = self.epsilon * (max_ - min_)
            scaled_noise = scaled_epsilon * noise

            #print(scaled_noise.shape)

            theta = x + scaled_noise
            
            if self.clip:
                theta = np.clip(theta, min_, max_)
                scaled_noise = theta - x

            is_zero = np.less(scaled_noise, 1e-6)
            scaled_noise[is_zero] = self.epsilon

            scores = pred_fn(theta)

            #print(scores.shape)
            assert scores.shape[0] == 2

            difference = np.array(scores[0] - scores[1])

            gradient_estimate = np.broadcast_to(difference, scaled_noise[0].shape) / (2 * scaled_noise[0])

            assert gradient_estimate.shape == x.shape
            gradient_estimates.append(gradient_estimate)

        return np.average(gradient_estimates, axis=0)

# Togliere
class GaussianSamplingDetectorEstimator____(DetectorEstimator):
    def __init__(self, epsilon, count, clip=True):
        self.epsilon = epsilon
        self.count = count
        self.clip = clip

    def _get_noise(self, shape, dtype):
        noise_shape = [self.count] + list(shape)
        noise = np.random.normal(size=noise_shape).astype(np.float32)

        return noise

    def gradient_one(self, x, detector):
        noise = self._get_noise(x.shape, x.dtype)

        min_, max_ = detector.clip_values
        scaled_epsilon = self.epsilon * (max_ - min_)
        scaled_noise = scaled_epsilon * noise

        theta = x + scaled_noise
        if self.clip:
            theta = np.clip(theta, min_, max_)
            scaled_noise = theta - x

        scores = detector.predict(theta)

        # TODO: Finire/correggere
        gradient_estimates = scores / scaled_noise

        assert gradient_estimates.shape == scaled_noise.shape

        # Calcola il gradiente medio
        average_gradient = np.average(gradient_estimates, axis=0)

        assert average_gradient.shape == x.shape