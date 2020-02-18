import numpy as np

from art.classifiers.classifier import Classifier, ClassifierNeuralNetwork, ClassifierGradients

class CounterAttackDetector(ClassifierNeuralNetwork, ClassifierGradients, Classifier):
    def __init__(self, target_classifier, attack, p, threshold, detector_estimator):
        super().__init__(clip_values=target_classifier.clip_values,
                                                  channel_index=target_classifier.channel_index,
                                                  defences=target_classifier.defences,
                                                  preprocessing=target_classifier.preprocessing)
        self.target_classifier = target_classifier
        self.attack = attack
        self.p = p
        self.threshold = threshold
        self.detector_estimator = detector_estimator

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        raise NotImplementedError

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        adversarials = self.attack.generate(x, batch_size=batch_size)
        differences = x - adversarials
        differences = differences.reshape((x.shape[0], -1))
        
        return np.linalg.norm(differences, ord=self.p, axis=1) - self.threshold

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    def nb_classes(self):
        return 1

    @property
    def input_shape(self):
        return self.target_classifier.input_shape

    @property
    def clip_values(self):
        return self.target_classifier.clip_values

    @property
    def channel_index(self):
        return self.target_classifier.channel_index

    @property
    def learning_phase(self):
        return self.target_classifier._learning_phase

    # Stima del gradiente usando Coordinate-Wise
    def class_gradient(self, x, label=None, **kwargs):
        assert label is None or label == 0

        return self.detector_estimator.gradient(x, self.predict, self.clip_values)

    def loss_gradient(self, x, y, **kwargs):
        raise NotImplementedError

    def get_activations(self, x, layer, batch_size):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`. This function is not supported for this detector.

        :raises: `NotImplementedException`
        """
        raise NotImplementedError

    def set_learning_phase(self, train):
        raise NotImplementedError

    def save(self, filename, path=None):
        raise NotImplementedError