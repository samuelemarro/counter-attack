import numpy as np

# Calcola il gradiente dello score rispetto agli input
# Le tecniche standard (incluso substitute.gradient()) calcolano
# il gradiente della Cross-Entropy Loss
class DetectorEstimator:
    def gradient(self, inputs):
        raise NotImplementedError()

    def gradient_one(self, x):
        raise NotImplementedError()

# Gradient originale restituisce il gradiente rispetto alla Cross-Entropy!
class SubstituteDetectorEstimator(DetectorEstimator):
    def __init__(self, substitute_model):
        self.substitute_model = substitute_model

    def gradient(self, inputs):
        return self.substitute_model.gradient(inputs)

    def gradient_one(self, x):
        return self.substitute_model.gradient_one(x)

# Variante di CoordinateWiseGradientEstimator che calcola la derivata
# dello score
class CoordinateWiseDetectorEstimator(DetectorEstimator):
    def __init__(self, detector, epsilon, clip=True):
        self.detector = detector
        self.epsilon = epsilon
        self.clip = clip

    def _get_noise(self, shape, dtype):
        N = np.prod(shape)
        noise = np.eye(N, N, dtype=dtype)
        noise = noise.reshape((N,) + shape)
        noise = np.concatenate([noise, -noise])
        return noise

    def gradient(self, inputs):
        gradients = []
        for x in inputs:
            gradients.append(self.gradient_one(x))
        gradients = np.array(gradients)

        return gradients

    def gradient_one(self, x):
        noise = self._get_noise(x.shape, x.dtype)
        N = len(noise)

        min_, max_ = self.detector.bounds()
        scaled_epsilon = self.epsilon * (max_ - min_)

        theta = x + scaled_epsilon * noise
        if self.clip:
            theta = np.clip(theta, min_, max_)
        loss = self.detector(theta)
        assert loss.shape == (N,)

        loss = loss.reshape((N,) + (1,) * x.ndim)
        assert loss.ndim == noise.ndim
        gradient = np.sum(loss * noise, axis=0)
        gradient /= 2 * scaled_epsilon
        return gradient
