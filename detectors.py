import torch

import utils

class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()


# TODO: Come si deve comportare CA in caso di fallimento???

class CounterAttackDetector(Detector):
    def __init__(self, attack, p):
        super().__init__()
        self.attack = attack
        self.p = p

    def forward(self, x):
        x = x.detach().clone() # Importante, altrimenti rischia di portare i gradienti dell'attacco fuori dal suo contesto
        with torch.enable_grad():
            # Nota l'assenza di y=
            adversarials = self.attack.perturb(x)
        adversarials = adversarials.detach()
        
        distances = utils.adversarial_distance(x, adversarials, self.p)

        # Distanza alta = bassa probabilità che sia un adversarial
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
        assert final_predictions.shape == (x.shape[0], predictions.shape[1] + 1)

        return final_predictions

class NormalisedDetectorModel(torch.nn.Module):
    """
    Appends the detector score to each prediction,
    adding a "rejected" class. It also performs
    normalisation following https://arxiv.org/abs/1705.07263
    """

    def __init__(self, model, detector, threshold):
        super().__init__()
        self.model = model
        self.detector = detector
        self.threshold = threshold

    def forward(self, x):
        predictions = self.model(x)
        scores = self.detector(x)

        assert scores.shape == (len(x),)

        # Accetta (positivo) se sono maggiori del threshold
        scores = scores - self.threshold

        # Scala
        prediction_max = predictions.max(1).values
        scores = (scores + 1) * prediction_max
        scores = scores.unsqueeze(1)

        final_predictions = torch.cat([predictions, scores], 1)
        
        assert len(final_predictions) == len(x)
        assert final_predictions.shape == (x.shape[0], predictions.shape[1] + 1)

        return final_predictions