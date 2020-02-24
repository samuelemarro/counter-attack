import gzip
import json
import logging
import pathlib
import pickle


import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

def save_zip(object, path, protocol=0):
    """
    Saves a compressed object to disk.
    """
    # Create the folder, if necessary
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    file = gzip.GzipFile(path, 'wb')
    pickled = pickle.dumps(object, protocol)
    file.write(pickled)
    file.close()

def load_zip(path):
    """
    Loads a compressed object from disk.
    """
    file = gzip.GzipFile(path, 'rb')
    buffer = b""
    while True:
        data = file.read()
        if data == b"":
            break
        buffer += data
    obj = pickle.loads(buffer)
    file.close()
    return obj

class AttackConfig:
    def __init__(self, config_dict):
        self.config_dict = config_dict
    def get_arguments(self, attack_name, domain, p, attack_type):
        kwargs = {}

        def load_kwargs(new_kwargs):
            for key, value in new_kwargs.items():
                if key in kwargs.keys():
                    logger.debug('Overriding value "{}" by replacing {} with {}.'.format(key, kwargs[key], value))
                kwargs[key] = value

        def loop_across_dict(current_dict, ramifications):
            if len(ramifications) == 0:
                load_kwargs(current_dict)
            else:
                current_ramification = ramifications[0]

                for branch in current_ramification:
                    if branch in current_dict.keys():
                        loop_across_dict(current_dict[branch], ramifications[1:])

        # The specific value overrides the general one
        loop_across_dict(self.config_dict, 
        [
            ['all_attacks', attack_name],
            ['all_domains', domain],
            ['all_distances', p],
            ['all_types', attack_type]
        ]
        )

        return kwargs

def read_attack_config_file(path):
    with open(path, 'r') as f:
        config_dict = json.load(f)

    return AttackConfig(config_dict)

def adversarial_distance(genuines, adversarials, p):
    assert genuines.shape == adversarials.shape

    if len(genuines) == 0:
        return torch.zeros([0])
    else:
        genuines = genuines.reshape(len(genuines), -1)
        adversarials = adversarials.reshape(len(adversarials), -1)

        distances = torch.nn.functional.pairwise_distance(genuines, adversarials, p)

        assert len(distances) == len(genuines)

        return distances

def remove_misclassified(model, images, labels):
    """Removes samples that are misclassified by the model.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model that might misclassify the images.
    images : torch.FloatTensor
        The images that might be misclassified.
    labels : torch.LongTensor
        The correct labels for the images.
    
    Returns
    -------
    tuple
        A tuple containing the images that are classified
        correctly and the corresponding labels.
    """
    predictions = model(images)
    predicted_labels = torch.argmax(predictions, axis=-1)

    assert predicted_labels.shape == labels.shape
    
    correct_label = torch.eq(predicted_labels, labels)

    return images[correct_label], labels[correct_label]

def check_success(model, images, labels, adversarials, has_detector):
    assert images.shape == adversarials.shape
    assert len(images) == len(labels)

    adversarial_predictions = model(adversarials)
    adversarial_labels = torch.argmax(adversarial_predictions, axis=1)
    
    assert adversarial_labels.shape == labels.shape

    successful = torch.logical_not(torch.eq(adversarial_labels, labels))

    if has_detector:
        num_classes = adversarial_predictions.shape[1]
        rejected = torch.eq(adversarial_labels, num_classes - 1)
        accepted = torch.logical_not(rejected)

        successful = successful & accepted

    return successful

# AdverTorch come considera i failed?
# Nota: Se l'originale viene rifiutato ma l'adversarial no, l'adversarial conta
# come successo anche se ha mantenuto la stessa label di partenza
# Testare!
def remove_failed(model, images, labels, adversarials, has_detector):
    successful = check_success(model, images, labels, adversarials, has_detector)
    
    return images[successful], labels[successful], adversarials[successful]

# TODO: Usarlo in altri posti?
def get_labels(model, images):
    return torch.argmax(model(images.to(model.device)), axis=1).to(images.device)

def show_images(images, adversarials, limit=None, model=None):
    if limit is not None:
        images = images[:limit]
        adversarials = adversarials[:limit]

    assert images.shape == adversarials.shape

    if model is None:
        labels = [None] * len(images)
        adversarial_labels = [None] * len(images)
    else:
        labels = get_labels(model, images)
        adversarial_labels = get_labels(model, adversarials)

    for image, label, adversarial, adversarial_label in zip(images, labels, adversarials, adversarial_labels):
        image_title = 'Original'
        adversarial_title = 'Adversarial'

        if model is not None:
            image_title += ' (label: {})'.format(label)
            adversarial_title += ' (label: {})'.format(adversarial_label)

        image = np.moveaxis(image.cpu().numpy(), 0, 2)
        adversarial = np.moveaxis(adversarial.cpu().numpy(), 0, 2)

        _, axes = plt.subplots(1, 2, squeeze=False)
        axes[0, 0].title(image_title)
        axes[0, 0].imshow(image)
        axes[0, 1].title(adversarial_title)
        axes[0, 1].imshow(adversarial)
        plt.show()