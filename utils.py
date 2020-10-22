import gzip
import json
import hashlib
import itertools
import logging
import os
import pathlib
import pickle
import sys

import advertorch
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
        return torch.zeros([0], device=genuines.device)
    else:
        genuines = genuines.flatten(1)
        adversarials = adversarials.flatten(1)

        distances = torch.nn.functional.pairwise_distance(genuines, adversarials, p)

        assert len(distances) == len(genuines)

        return distances

def one_many_adversarial_distance(one, many, p):
    assert one.shape == many.shape[1:]

    # Add a batch dimension that matches many's batch size
    one = one.unsqueeze(0).expand(len(many), -1, -1, -1)

    assert one.shape == many.shape

    return adversarial_distance(one, many, p)

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
    predicted_labels = get_labels(model, images)

    assert predicted_labels.shape == labels.shape
    
    correct_label = torch.eq(predicted_labels, labels)

    return images[correct_label], labels[correct_label]

def misclassified(model, adversarials, labels, has_detector):
    assert len(adversarials) == len(labels)

    adversarial_predictions = model(adversarials)
    
    return misclassified_outputs(adversarial_predictions, labels, has_detector)

def misclassified_outputs(outputs, labels, has_detector):
    assert len(outputs) == len(labels)

    adversarial_labels = torch.argmax(outputs, axis=1)
    
    assert adversarial_labels.shape == labels.shape

    successful = torch.logical_not(torch.eq(adversarial_labels, labels))

    if has_detector:
        num_classes = outputs.shape[1]
        rejected = torch.eq(adversarial_labels, num_classes - 1)
        accepted = torch.logical_not(rejected)

        successful = successful & accepted

    return successful

def early_rejection(x, adversarials, labels, adversarial_output, p, threshold, targeted):
    predicted_labels = torch.argmax(adversarial_output, dim=1)

    if targeted:
        successful = torch.eq(labels, predicted_labels)
    else:
        successful = ~torch.eq(labels, predicted_labels)
    
    distances = adversarial_distance(x, adversarials, p)
    valid_distance = distances < threshold

    return successful & valid_distance

# AdverTorch come considera i failed?
# Nota: Se l'originale viene rifiutato ma l'adversarial no, l'adversarial conta
# come successo anche se ha mantenuto la stessa label di partenza
# Testare!
def remove_failed(model, images, labels, adversarials, has_detector, p=None, eps=None):
    assert len(images) == len(labels)
    assert len(images) == len(adversarials)

    successful = misclassified(model, adversarials, labels, has_detector)

    if eps is not None:
        assert p is not None
        distances = adversarial_distance(images, adversarials, p)

    adversarials = list(adversarials)

    for i in range(len(images)):
        valid_distance = (eps is None) or (distances[i] < eps)
        if not successful[i] and valid_distance:
            adversarials[i] = None
    
    return adversarials


# Returns b if filter_ is True, else a
def fast_boolean_choice(a, b, filter_):
    filter_shape = [1] + list(a.shape)[1:]
    reshaped_filter = filter_.reshape(filter_shape).float()
    repeat_dimensions = [len(a)] + (len(a.shape) - 1) * [1]

    repeated_filter = reshaped_filter.repeat(repeat_dimensions)
    return a + repeated_filter * (b - a)

def get_labels(model, images):
    model_device = next(model.parameters()).device
    return torch.argmax(model(images.to(model_device)), axis=1).to(images.device)

def show_images(images, adversarials, limit=None, model=None):
    assert len(images) == len(adversarials)

    successful_images = []
    successful_adversarials = []

    for image, adversarial in zip(images, adversarials):
        if adversarial is not None:
            successful_images.append(image)
            successful_adversarials.append(adversarial)

    successful_images = torch.stack(successful_images)
    successful_adversarials = torch.stack(successful_adversarials)

    if limit is not None:
        successful_images = successful_images[:limit]
        successful_adversarials = successful_adversarials[:limit]

    assert successful_images.shape == successful_adversarials.shape

    if model is None:
        labels = [None] * len(successful_images)
        adversarial_labels = [None] * len(successful_images)
    elif len(successful_images) > 0:
        labels = get_labels(model, successful_images)
        adversarial_labels = get_labels(model, successful_adversarials)
    else:
        labels = []
        adversarial_labels = []

    for image, label, adversarial, adversarial_label in zip(successful_images, labels, successful_adversarials, adversarial_labels):
        image_title = 'Original'
        adversarial_title = 'Adversarial'

        if model is not None:
            image_title += ' (label: {})'.format(label)
            adversarial_title += ' (label: {})'.format(adversarial_label)

        if image.shape[0] == 1:
            plt.style.use('grayscale')

        normalisation = plt.Normalize(vmin=0, vmax=1)

        image = np.moveaxis(image.cpu().numpy(), 0, 2).squeeze()
        adversarial = np.moveaxis(adversarial.cpu().numpy(), 0, 2).squeeze()
        difference = np.abs(image - adversarial)

        _, axes = plt.subplots(1, 3, squeeze=False)
        axes[0, 0].title.set_text(image_title)
        axes[0, 0].imshow(image, norm=normalisation)
        axes[0, 1].title.set_text(adversarial_title)
        axes[0, 1].imshow(adversarial, norm=normalisation)
        axes[0, 2].title.set_text('Difference')
        axes[0, 2].imshow(difference, norm=normalisation)

        print('L2 norm: {}'.format(np.linalg.norm(difference)))
        print('LInf norm: {}'.format(np.max(difference)))

        plt.show()

def maybe_stack(tensors, fallback_shape, dtype=torch.float, device='cpu'):
    if len(tensors) > 0:
        return torch.stack(tensors)
    else:
        if fallback_shape is None:
            shape = (0, )
        else:
            shape = (0, ) + fallback_shape
        return torch.zeros(shape, dtype=dtype, device=device)

def tensor_md5(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor_content = tensor.tostring()

    return int(hashlib.md5(tensor).hexdigest(), 16)

def consistent_wrapper(linked_tensor, wrapped_function):
    # Save the current RNG state
    rng_state = torch.get_rng_state()

    # Get a consistent seed
    seed = tensor_md5(linked_tensor) % (2**63)

    torch.manual_seed(seed)

    random_tensor = wrapped_function()

    # Restore the RNG state
    torch.set_rng_state(rng_state)

    return random_tensor

def consistent_randint(linked_tensor, min, max, shape, device):
    return consistent_wrapper(linked_tensor,
    lambda: torch.randint(min, max, shape, device=device))

def consistent_rand_init_delta(deltas, x, ord, eps, clip_min, clip_max):
    assert len(x) == len(deltas)
    assert len(x) == len(eps)

    for i, (image, delta, image_eps) in enumerate(zip(x, deltas, eps)):
        delta = delta.unsqueeze(0)
        unsqueezed_image = image.unsqueeze(0)
        unsqueezed_eps = eps.unsqueeze(0)
        consistent_wrapper(image,
            lambda: advertorch.attacks.utils.rand_init_delta(delta, unsqueezed_image, ord, unsqueezed_eps, clip_min, clip_max)[0]
        )
    
    return deltas

class ConsistentGenerator:
    def __init__(self, wrapped_function):
        self.rng_state_dict = {}
        self.wrapped_function = wrapped_function

    def generate(self, tensor_id, linked_tensor):
        if torch.is_tensor(tensor_id):
            tensor_id = tensor_id.cpu().numpy()

        if isinstance(tensor_id, np.ndarray):
            tensor_id = tensor_id.item()

        # Save the current RNG state
        current_rng_state = torch.get_rng_state()

        if tensor_id in self.rng_state_dict:
            # Load an existing RNG state
            torch.set_rng_state(self.rng_state_dict[tensor_id])
        else:
            # Use the md5 hash as seed
            torch.manual_seed(tensor_md5(linked_tensor) % (2**63))

        return_value = self.wrapped_function()

        # Save the new RNG state for future use
        self.rng_state_dict[tensor_id] = torch.get_rng_state()

        # Restore the RNG state
        torch.set_rng_state(current_rng_state)

        return return_value

    def batch_generate(self, tensor_ids, linked_tensors):
        return_values = []
        for tensor_id, linked_tensor in zip(tensor_ids, linked_tensors):
            return_values.append(self.generate(tensor_id, linked_tensor))

        return torch.stack(return_values)

def create_label_dataset(model, images, batch_size):
    image_dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size)

    labels = []

    for image_batch in dataloader:
        # Convert to tensor
        image_batch = torch.stack(image_batch).squeeze(0)

        label_batch = get_labels(model, image_batch)
        labels += list(label_batch)

    labels = torch.stack(labels)

    return torch.utils.data.TensorDataset(images, labels)

def powerset(iterable, allow_empty=False):
    if allow_empty:
        start = 0
    else:
        start = 1

    s = list(iterable)

    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(start, len(s)+1)))  

class HiddenPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout