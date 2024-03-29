import logging

import numpy as np
import torch

import utils

logger = logging.getLogger(__name__)


class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class CounterAttackDetector(Detector):
    def __init__(self, attack, model, p):
        super().__init__()
        self.attack = attack
        self.model = model
        self.p = p

    def forward(self, x):
        x = x.detach().clone()
        with torch.enable_grad():
            adversarials = self.attack.perturb(x).detach()
        adversarials = adversarials.detach()

        assert len(adversarials) == len(x)

        distances = utils.adversarial_distance(x, adversarials, self.p)

        assert len(distances) == len(x)

        labels = utils.get_labels(self.model, x).detach()
        successful = utils.misclassified(
            self.model, adversarials, labels, False).detach()

        distances[~successful] = 10000

        return -distances


class StandardDetectorModel(torch.nn.Module):
    """
    Appends the detector store to each prediction,
    adding a "rejected" class.
    """

    def __init__(self, model, detector):
        super().__init__()
        self.model = model
        self.detector = detector

    def forward(self, x):
        predictions = self.model(x)
        scores = self.detector(x)

        assert scores.shape == (len(x),)

        scores = scores.unsqueeze(1)

        final_predictions = torch.cat([predictions, scores], 1)

        assert len(final_predictions) == len(x)
        assert final_predictions.shape == (
            x.shape[0], predictions.shape[1] + 1)

        return final_predictions


class NormalisedDetectorModel(torch.nn.Module):
    """
    Appends the detector score to each prediction,
    adding a "rejected" class. The score is
    normalised following https://arxiv.org/abs/1705.07263
    """

    def __init__(self, model, detector, threshold):
        """Initialises the NormalisedDetectorModel

        Parameters
        ----------
        model : torch.nn.Module
            The undefended classifier.
        detector : torch.nn.Module
            The detector.
        threshold : float
            If a score is above the threshold, the
            corresponding sample is considered
            rejected.
        """
        super().__init__()
        logger.debug(f'Creating detector with threshold {threshold}.')
        self.model = model
        self.detector = detector
        self.threshold = threshold

    def forward(self, x):
        predictions = self.model(x)
        scores = self.detector(x)

        assert scores.shape == (len(x),)

        # Accept (positive score) if the detector score
        # is bigger than the threshold
        scores = scores - self.threshold

        # Scale the output
        prediction_max = predictions.max(1).values
        scores = (scores + 1) * prediction_max
        scores = scores.unsqueeze(1)

        final_predictions = torch.cat([predictions, scores], 1)

        assert len(final_predictions) == len(x)
        assert final_predictions.shape == (
            x.shape[0], predictions.shape[1] + 1)

        return final_predictions


class DetectorPool(Detector):
    def __init__(self, detectors, p):
        super().__init__()
        self.detectors = detectors
        self.p = p

    def forward(self, x):
        detector_scores = torch.stack([detector(x)
                                       for detector in self.detectors])

        assert detector_scores.shape == (len(self.detectors), len(x))

        # Pick the highest for each element of the batch
        highest_scores, _ = torch.max(detector_scores, 0)

        assert highest_scores.shape == (len(x),)

        return highest_scores
