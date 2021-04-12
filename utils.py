import gzip
import itertools
import json
import logging
import pathlib
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_zip(obj, path, protocol=0):
    """
    Saves a compressed object to disk.
    """
    # Create the folder, if necessary
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    file = gzip.GzipFile(path, 'wb')
    pickled = pickle.dumps(obj, protocol)
    file.write(pickled)
    file.close()


def load_zip(path):
    """
    Loads a compressed object from disk.
    """
    file = gzip.GzipFile(path, 'rb')
    buffer = b''
    while True:
        data = file.read()
        if data == b'':
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
                    logger.debug(
                        'Overriding key %s by replacing %s with %s.',
                        key, kwargs[key], value)
                else:
                    logger.debug('Registering %s=%s.', key, value)

                kwargs[key] = value

        def loop_across_dict(current_dict, selectors):
            if 'params' in current_dict:
                logger.debug('Found params.')
                load_kwargs(current_dict['params'])

            if len(selectors) == 0:
                return

            general_selector, specific_selector = selectors[0]

            if general_selector in current_dict and specific_selector in current_dict:
                raise RuntimeError('Both selectors available: cannot choose.')

            if specific_selector in current_dict:
                logger.debug('Going into %s.', specific_selector)
                loop_across_dict(current_dict[specific_selector], selectors[1:])
            elif general_selector in current_dict:
                assert len(current_dict.keys()) <= 2
                logger.debug('Going into %s.', general_selector)
                loop_across_dict(current_dict[general_selector], selectors[1:])

        # The specific value overrides the general one, from outermost to innermost
        loop_across_dict(self.config_dict,
                         [
                             ('all_attacks', attack_name),
                             ('all_domains', domain),
                             ('all_distances', p),
                             ('all_types', attack_type)
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
        return torch.zeros([0.], device=genuines.device)
    else:
        # pairwise_distance only accepts 2D tensors
        genuines = genuines.flatten(1)
        adversarials = adversarials.flatten(1)

        distances = torch.nn.functional.pairwise_distance(
            genuines, adversarials, p)
        assert distances.shape == (len(genuines),)

        return distances

def one_many_adversarial_distance(one, many, p):
    assert one.shape == many.shape[1:]

    # Add a batch dimension that matches many's batch size
    one = one.unsqueeze(0).expand(len(many), -1, -1, -1)
    assert one.shape == many.shape

    return adversarial_distance(one, many, p)


def apply_misclassification_policy(model, images, true_labels, policy):
    if policy == 'ignore':
        return images, true_labels, true_labels
    else:
        predicted_labels = get_labels(model, images)
        assert predicted_labels.shape == true_labels.shape

        if policy == 'remove':
            correct_label = torch.eq(predicted_labels, true_labels)
            return images[correct_label], true_labels[correct_label], true_labels[correct_label]
        elif policy == 'use_predicted':
            return images, true_labels, predicted_labels
        else:
            raise NotImplementedError(f'Unsupported policy "{policy}".')

def check_successful(model, adversarials, labels, targeted):
    assert len(adversarials) == len(labels)

    adversarial_outputs = model(adversarials)

    adversarial_labels = torch.argmax(adversarial_outputs, axis=1)
    assert adversarial_labels.shape == labels.shape

    if targeted:
        return torch.eq(adversarial_labels, labels)
    else:
        return ~torch.eq(adversarial_labels, labels)


def misclassified(model, adversarials, labels, has_detector):
    assert len(adversarials) == len(labels)

    adversarial_predictions = model(adversarials)
    assert len(adversarial_predictions) == len(adversarials)

    return misclassified_outputs(adversarial_predictions, labels, has_detector)


def misclassified_outputs(outputs, labels, has_detector):
    assert len(outputs) == len(labels)

    adversarial_labels = torch.argmax(outputs, axis=1)
    assert adversarial_labels.shape == labels.shape

    successful = ~torch.eq(adversarial_labels, labels)

    if has_detector:
        num_classes = outputs.shape[1]
        rejected = torch.eq(adversarial_labels, num_classes - 1)

        successful = successful & ~rejected

    return successful

def remove_failed(model, images, labels, adversarials, has_detector):
    assert len(images) == len(labels)
    assert images.shape == adversarials.shape

    successful = misclassified(model, adversarials, labels, has_detector)
    assert successful.shape == (len(images),)

    adversarials = list(adversarials)

    for i in range(len(images)):
        if not successful[i]:
            adversarials[i] = None

    return adversarials

# Returns b if filter_ is True, else a
def fast_boolean_choice(a, b, filter_, reshape=True):
    assert len(a) == len(b) == len(filter_)

    if reshape:
        pre_expansion_shape = [len(filter_)] + ([1] * (len(a.shape) - 1))
        filter_ = filter_.reshape(*pre_expansion_shape)

        post_expansion_shape = [len(filter_)] + list(a.shape[1:])
        filter_ = filter_.expand(*post_expansion_shape)

    filter_ = filter_.float()

    return filter_ * b + (1 - filter_) * a


def get_labels(model, images):
    model_device = next(model.parameters()).device
    outputs = model(images.to(model_device))
    assert len(outputs) == len(images)

    return torch.argmax(outputs, axis=1).to(images.device)


def replace_active(from_, to, active, filter_):
    assert len(to) == len(active)
    assert len(from_) == len(filter_)

    replace_to = active.clone()
    replace_to[active] = filter_
    to[replace_to] = from_[filter_]


def show_images(images, adversarials, limit=None, model=None):
    try:
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

        assert len(successful_images) == len(labels) == len(adversarial_labels)

        for image, label, adversarial, adversarial_label in zip(successful_images, labels, successful_adversarials, adversarial_labels):
            image_title = 'Original'
            adversarial_title = 'Adversarial'

            if model is not None:
                image_title += f' (label: {label})'
                adversarial_title += f' (label: {adversarial_label})'

            if image.shape[0] == 1:
                # Use grayscale for images with only one channel
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

            print(f'L2 norm: {np.linalg.norm(difference.flatten())}')
            print(f'LInf norm: {np.max(difference)}')

            plt.show()
    except Exception as e:
        # Never let a visualization error cause an exception
        logger.error('Failed to show images: %s.', e)


def maybe_stack(tensors, fallback_shape, dtype=torch.float, device='cpu'):
    if len(tensors) > 0:
        return torch.stack(tensors)
    else:
        if fallback_shape is None:
            shape = (0, )
        else:
            shape = [0] + list(fallback_shape)
        return torch.zeros(shape, dtype=dtype, device=device)

def clip_adversarial(adversarial, genuine, epsilon, input_min=0, input_max=1):
    # Note: Supports both single and batch modes

    assert adversarial.shape == genuine.shape

    clipped_lower = torch.clip(genuine - epsilon, min=input_min, max=input_max)
    clipped_upper = torch.clip(genuine + epsilon, min=input_min, max=input_max)

    replace_lower = adversarial < clipped_lower
    replace_upper = adversarial > clipped_upper

    # Clip to [clipped_lower, clipped_upper]
    adversarial = fast_boolean_choice(adversarial, clipped_lower, replace_lower, reshape=False)
    adversarial = fast_boolean_choice(adversarial, clipped_upper, replace_upper, reshape=False)

    # Note: Technically the additional clip is unnecessary and is only used as a safety measure
    return torch.clip(adversarial, min=input_min, max=input_max)

def create_label_dataset(model, images, batch_size):
    image_dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=batch_size)

    labels = []

    for image_batch in dataloader:
        # Convert to tensor
        image_batch = torch.stack(image_batch).squeeze(0)

        label_batch = get_labels(model, image_batch)
        labels += list(label_batch)

    labels = torch.stack(labels)

    return torch.utils.data.TensorDataset(images, labels)


def powerset(iterable, allow_empty):
    if allow_empty:
        start = 0
    else:
        start = 1

    s = list(iterable)

    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(start, len(s)+1)))

def set_seed(seed):
    logger.info('Setting seed.')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    pytorch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()

    return python_state, numpy_state, pytorch_state, cuda_states

def set_rng_state(state_info):
    python_state, numpy_state, pytorch_state, cuda_states = state_info

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(pytorch_state)
    torch.cuda.set_rng_state_all(cuda_states)

def enable_determinism():
    logger.info('Enabling determinism.')
    torch.backends.cudnn.benchmark = False
    torch.set_deterministic(True)
