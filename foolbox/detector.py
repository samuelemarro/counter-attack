import foolbox

class Detector(foolbox.models.Model):
    def num_classes(self):
        return 1
    

class ModelWithDetector(foolbox.models.ModelWrapper):
    def __init__(self, model, detector):
        super().__init__(model)
        self.detector = detector

    def forward(self, inputs):
        return zip(self.wrapped_model.forward(inputs), self.detector(inputs))

    def forward_one(self, x):
        return (self.wrapped_model.forward_one(x), self.detector.forward_one(x))

# Usiamo i gradienti veri del modello e i gradienti stimati del detector
# Nota: Backward() restituisce solo rispetto al modello
class ModelWithDetectorEstimatedGradients(ModelWithDetector, foolbox.models.DifferentiableModel):
    def __init__(self, model, detector, detector_estimator):
        super().__init__(model, detector)
        self.detector_estimator = detector_estimator

    def forward_and_gradient(self, inputs, label):
        model_forward, model_gradient = self.wrapped_model.forward_and_gradient(inputs, label)
        detector_forward = self.detector.forward(inputs, label)
        detector_gradient = self.detector_estimator.gradient(inputs)
        
        return zip(model_forward, detector_forward, model_gradient, detector_gradient)
    
    def forward_and_gradient_one(self, x, label):
        model_forward, model_gradient = self.wrapped_model.forward_and_gradient_one(x, label)
        detector_forward = self.detector.forward_one(x, label)
        detector_gradient = self.detector_estimator.gradient_one(x)
        
        return (model_forward, detector_forward, model_gradient, detector_gradient)

    def gradient(self, inputs, labels):
        return zip(self.wrapped_model.gradient(inputs, labels), self.detector_estimator.gradient(inputs))

    def gradient_one(self, x, label):
        return (self.wrapped_model.gradient_one(x, label), self.detector_estimator.gradient_one(x))

    def backward(self, gradient, inputs):
        return self.wrapped_model.backward(gradient, inputs)

    def backward_one(self, gradient, x):
        return self.wrapped_model.backward_one(gradient, x)

        