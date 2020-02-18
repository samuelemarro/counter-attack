import numpy as np
import detector

class CounterAttackDetector(detector.Detector):
    def __init__(self, bounds, channel_axis, attack, p, preprocessing=(0, 1)):
        super().__init__(bounds, channel_axis, preprocessing)
        self.attack = attack
        self.p = p

    def forward(self, inputs):
        adversarials = self.attack(inputs)
        differences = inputs - adversarials
        differences = differences.reshape((inputs.shape[0], -1))
        
        return np.linalg.norm(differences, ord=self.p, axis=1)