import copy

import torch
import numpy as np

import detectors
import torch_utils

def test_gradient_estimation(model, estimator_fn, images, *args, output_threshold=None, gradient_threshold=None):
    print('====Gradient Estimation Test=====')
    images = images.clone()
    images.requires_grad = True
    images = images.double()
    model = copy.deepcopy(model).double()
    standard_outputs = model(images)
    standard_loss = torch.sum(standard_outputs ** 2)
    standard_gradients = torch.autograd.grad(standard_loss, images)[0]

    #estimated_outputs = torch_estimation.BinomialSamplingEstimator_F.apply(images, model, 1e-4, 3000)
    estimated_outputs = estimator_fn.apply(images, model.forward, *args)
    estimated_loss = torch.sum(estimated_outputs ** 2)
    estimated_gradients = torch.autograd.grad(estimated_loss, images)[0]

    # Converti in numpy
    images = images.cpu().detach().numpy()
    standard_outputs = standard_outputs.cpu().detach().numpy()
    estimated_outputs = estimated_outputs.cpu().detach().numpy()
    standard_gradients = standard_gradients.cpu().detach().numpy()
    estimated_gradients = estimated_gradients.cpu().detach().numpy()

    assert standard_gradients.shape == estimated_gradients.shape

    #print('Standard: {}'.format(standard_gradients))
    #print('Estimated: {}'.format(estimated_gradients))

    output_difference = np.average(np.abs(standard_outputs - estimated_outputs))
    gradient_difference = np.average(np.abs(standard_gradients - estimated_gradients))

    relative_output_difference = output_difference / np.average(np.abs(standard_outputs))
    relative_gradient_difference = gradient_difference / np.average(np.abs(standard_gradients))

    print('Output difference: {:.2e} (relative: {:.3f}%)'.format(output_difference, relative_output_difference * 100.0))
    print('Gradient difference: {:.2e} (relative: {:.3f}%)'.format(gradient_difference, relative_gradient_difference * 100.0))

    if output_threshold is not None:
        assert relative_output_difference <= output_threshold

    if gradient_threshold is not None:
        assert relative_gradient_difference <= gradient_threshold

def test_detector_output(model, detector, threshold, images, predictions_threshold=None, detector_threshold=None, reject_threshold=None):
    print('====Detector Output Test=====')
    standard_model = detectors.StandardDetectorModel(model, detector)
    normalised_model = detectors.NormalisedDetectorModel(model, detector, threshold)

    standard_output = standard_model(images).detach().cpu().numpy()
    standard_score = standard_output[:, -1]
    standard_predictions = standard_output[:, :-1]

    normalised_output = normalised_model(images).detach().cpu().numpy()
    normalised_score = normalised_output[:, -1]
    normalised_predictions = normalised_output[:, :-1]

    detector_score = detector(images).detach().cpu().numpy()

    assert standard_output.shape == normalised_output.shape

    print('Standard output: {}'.format(standard_output))
    print('Normalised output: {}'.format(normalised_output))
    print('Detector output: {}'.format(detector_score))

    predictions_difference = np.average(np.abs(standard_predictions - normalised_predictions))
    relative_predictions_difference = predictions_difference / np.average(np.abs(standard_predictions))

    print('Predictions difference: {} (relative: {:.3f}%)'.format(predictions_difference, relative_predictions_difference * 100.0))

    if predictions_threshold is not None:
        assert relative_predictions_difference <= predictions_threshold

    score_difference = np.average(np.abs(standard_score - detector_score))
    relative_score_difference = score_difference / np.average(np.abs(standard_score))

    print('Score difference: {} (relative: {:.3f}%)'.format(score_difference, relative_score_difference * 100.0))

    if detector_threshold is not None:
        assert relative_score_difference <= detector_threshold

    standard_accept = standard_score >= threshold
    normalised_accept = normalised_score >= 0

    rejection_difference = np.count_nonzero(np.logical_not(np.equal(standard_accept, normalised_accept)))
    rejection_difference_rate = rejection_difference / len(standard_accept)

    print('Rejection differences: {} (rate: {:.3f}%)'.format(rejection_difference, rejection_difference_rate * 100.0))

    if reject_threshold is not None:
        assert rejection_difference_rate <= reject_threshold

def test_model_batch_limiting(model, images, batch_size, predictions_threshold=None):
    print('====Model Batch Limiting Test=====')
    batch_limited_model = torch_utils.BatchLimitedModel(model, batch_size)

    standard_predictions = model(images).detach().cpu().numpy()
    batch_limited_predictions = batch_limited_model(images).detach().cpu().numpy()

    predictions_difference = np.average(np.abs(standard_predictions - batch_limited_predictions))
    relative_predictions_difference = predictions_difference / np.average(np.abs(standard_predictions))

    print('Predictions difference: {} (relative: {:.3f}%)'.format(predictions_difference, relative_predictions_difference * 100.0))

    if predictions_threshold is not None:
        assert relative_predictions_difference <= predictions_threshold


def test_normalisation(model, images, predictions_threshold=None):
    print('====Normalisation Test=====')
    mean = np.array([0., 0., 0.])
    stdevs = np.array([1., 1., 1.])
    normalised_model = torch.nn.Sequential(torch_utils.Normalisation(mean, stdevs), model)

    standard_predictions = model(images).detach().cpu().numpy()
    normalised_predictions = normalised_model(images).detach().cpu().numpy()

    predictions_difference = np.average(np.abs(standard_predictions - normalised_predictions))
    relative_predictions_difference = predictions_difference / np.average(np.abs(standard_predictions))

    print('Predictions difference: {} (relative: {:.3f}%)'.format(predictions_difference, relative_predictions_difference * 100.0))

    if predictions_threshold is not None:
        assert relative_predictions_difference <= predictions_threshold