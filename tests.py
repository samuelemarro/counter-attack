import logging
from pathlib import Path
import time

import torch
from tqdm import tqdm

import detectors
import utils
import adversarial_dataset

logger = logging.getLogger(__name__)

# TODO: A volte attack_config è prima di generation_kwargs, a volte dopo
SIMILARITY_THRESHOLD = 1e-6

def accuracy(model, loader, device):
    correct_count = 0
    total_count = 0

    model.to(device)

    for images, true_labels in tqdm(loader, desc='Accuracy Test'):
        total_count += len(images)
        images = images.to(device)
        true_labels = true_labels.to(device)

        predicted_labels = utils.get_labels(model, images).detach()

        correct = torch.eq(predicted_labels, true_labels)
        correct_count += len(torch.nonzero(correct))

    return correct_count / total_count


def attack_test(model, attack, loader, p, misclassification_policy, device, attack_configuration, generation_kwargs, start, stop, defended_model, blind_trust=False):
    if attack.targeted:
        raise NotImplementedError('Targeted attack tests are not supported.')

    logger.debug('Misclassification policy: %s.', misclassification_policy)
    logger.debug('Blind trust: %s', blind_trust)

    if misclassification_policy == 'remove':
        logger.warning('Remember that using "remove" as a misclassification policy can interfere with dataset merging.')

    model.to(device)

    all_images = []
    all_labels = []
    all_true_labels = []
    all_adversarials = []

    for images, true_labels in tqdm(loader, desc='Attack Test'):
        images = images.to(device)
        true_labels = true_labels.to(device)

        assert len(images) == len(true_labels)

        images, true_labels, labels = utils.apply_misclassification_policy(
            model, images, true_labels, misclassification_policy)

        if len(images) == 0:
            assert misclassification_policy == 'remove'
            logger.warning('0 images left after removing misclassified, skipping batch.')
            continue

        adversarials = attack.perturb(images, y=labels).detach()
        assert adversarials.shape == images.shape

        if blind_trust:
            adversarials = list(adversarials)
        else:
            if defended_model is None:
                adversarials = utils.remove_failed(
                    model, images, labels, adversarials, False)
            else:
                adversarials = utils.remove_failed(
                    defended_model, images, labels, adversarials, True)

        # Move to CPU
        images = images.cpu()
        labels = labels.cpu()
        true_labels = true_labels.cpu()

        for i in range(len(adversarials)):
            if adversarials[i] is not None:
                adversarials[i] = adversarials[i].cpu()

        all_images += list(images)
        all_labels += list(labels)
        all_true_labels += list(true_labels)
        all_adversarials += list(adversarials)

    assert len(all_images) == len(all_labels)
    assert len(all_images) == len(all_true_labels)
    assert len(all_images) == len(all_adversarials)

    return adversarial_dataset.AdversarialDataset(all_images, all_labels, all_true_labels, all_adversarials, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs)


def mip_test(model, attack, loader, p, misclassification_policy, device, attack_configuration, generation_kwargs, start, stop, pre_adversarial_dataset=None, log_dir=None):
    test_start_timestamp = time.time()

    if attack.targeted:
        raise NotImplementedError('Targeted attack tests are not supported.')

    if misclassification_policy == 'remove':
        logger.warning('Remember that using "remove" as a misclassification policy can interfere with dataset merging.')

    model.to(device)

    all_images = []
    all_labels = []
    all_true_labels = []
    all_adversarials = []
    all_lower_bounds = []
    all_upper_bounds = []
    all_elapsed_times = []
    all_extra_infos = []

    test_loop_start_timestamp = time.time()

    for index, (images, true_labels) in tqdm(enumerate(loader), desc='MIP Test'):
        images = images.to(device)
        true_labels = true_labels.to(device)

        assert len(images) == len(true_labels)

        images, true_labels, labels = utils.apply_misclassification_policy(
            model, images, true_labels, misclassification_policy)

        if len(images) == 0:
            assert misclassification_policy == 'remove'
            logger.warning('0 images left after removing misclassified, skipping batch.')
            continue

        if pre_adversarial_dataset is None:
            pre_images = None
            pre_adversarials = None
        else:
            matching_indices = pre_adversarial_dataset.index_of_genuines(
                images)

            if any(i == -1 for i in matching_indices):
                raise RuntimeError('Could not find a matching element in the pre-adversarial dataset '
                                   'for a genuine. Check that the correct pre-adversarial set is being used.')

            pre_images = [pre_adversarial_dataset.genuines[i]
                          for i in matching_indices]
            pre_labels = [pre_adversarial_dataset.labels[i]
                         for i in matching_indices]
            pre_true_labels = [pre_adversarial_dataset.true_labels[i]
                         for i in matching_indices]
            pre_adversarials = [pre_adversarial_dataset.adversarials[i]
                                for i in matching_indices]

            assert len(pre_images) == len(labels) == len(true_labels) == len(images)
            assert len(pre_adversarials) == len(images)

            for i in range(len(pre_images)):
                assert pre_images[i].shape == images[i].shape
                pre_images[i] = pre_images[i].to(device)
                pre_labels[i] = pre_labels[i].to(device)
                pre_true_labels[i] = pre_true_labels[i].to(device)

                assert torch.eq(labels[i], pre_labels[i])
                assert torch.eq(true_labels[i], pre_true_labels[i])

                if pre_adversarials[i] is not None:
                    assert pre_adversarials[i].shape == images[i].shape
                    pre_adversarials[i] = pre_adversarials[i].to(device)

            # Check that the images are the same
            all_match = all([torch.max(torch.abs(image - pre_image))
                             < SIMILARITY_THRESHOLD for image, pre_image in zip(images, pre_images)])

            if not all_match:
                raise RuntimeError('The pre-adversarials refer to different genuines. '
                                   'This can slow down MIP at best and make it fail at worst. '
                                   'Check that the correct pre-adversarial dataset is being used.')

        adversarials, lower_bounds, upper_bounds, elapsed_times, extra_infos = attack.perturb_advanced(
            images, y=labels, starting_points=pre_adversarials, log_dir=None if log_dir is None else Path(log_dir) / f'batch_{index}')

        assert len(adversarials) == len(images)
        assert len(adversarials) == len(lower_bounds)
        assert len(adversarials) == len(upper_bounds)
        assert len(adversarials) == len(elapsed_times)
        assert len(adversarials) == len(extra_infos)

        # Move to CPU
        images = images.cpu()
        labels = labels.cpu()
        true_labels = true_labels.cpu()
        adversarials = [None if adversarial is None else adversarial.detach().cpu() for adversarial in adversarials]

        all_images += list(images)
        all_labels += list(labels)
        all_true_labels += list(true_labels)
        all_adversarials += list(adversarials)
        all_lower_bounds += list(lower_bounds)
        all_upper_bounds += list(upper_bounds)
        all_elapsed_times += list(elapsed_times)
        all_extra_infos += list(extra_infos)

    test_loop_end_timestamp = time.time()

    assert len(all_images) == len(all_labels)
    assert len(all_images) == len(all_true_labels)
    assert len(all_images) == len(all_adversarials)
    assert len(all_images) == len(all_lower_bounds)
    assert len(all_images) == len(all_upper_bounds)
    assert len(all_images) == len(all_elapsed_times)

    test_end_timestamp = time.time()

    global_extra_info = {
        'times' : {
            'mip_test' : {
                'start_timestamp' : test_start_timestamp,
                'end_timestamp' : test_end_timestamp
            },
            'mip_test_loop' : {
                'start_timestamp' : test_loop_start_timestamp,
                'end_timestamp' : test_loop_end_timestamp
            }
        }
    }

    return adversarial_dataset.MIPDataset(all_images, all_labels, all_true_labels, all_adversarials, all_lower_bounds, all_upper_bounds, all_elapsed_times, all_extra_infos, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs, global_extra_info)


def multiple_evasion_test(model, test_names, attacks, defended_models, loader, p, misclassification_policy, device, attack_configuration, start, stop, generation_kwargs):
    assert all(not attack.targeted for attack in attacks)
    assert all(attack.predict == defended_model.predict for attack,
               defended_model in zip(attacks, defended_models))

    if misclassification_policy == 'remove':
        logger.warning('Remember that using "remove" as a misclassification policy can interfere with dataset merging.')

    model.to(device)

    for defended_model in defended_models:
        defended_model.to(device)

    assert len(test_names) == len(attacks)
    assert len(test_names) == len(defended_models)

    all_images = []
    all_true_labels = []
    all_attack_results = []

    for images, true_labels in tqdm(loader, desc='Multiple Evasion Test'):
        images = images.to(device)
        true_labels = true_labels.to(device)

        images, true_labels, labels = utils.apply_misclassification_policy(
            model, images, true_labels, misclassification_policy)

        if len(images) == 0:
            assert misclassification_policy == 'remove'
            logger.warning('0 images left after removing misclassified, skipping batch.')
            continue

        attack_results = [dict() for _ in range(len(images))]

        for test_name, attack, defended_model in zip(test_names, attacks, defended_models):
            # Nota y=labels
            adversarials = attack.perturb(images, y=labels).detach()

            assert adversarials.shape == images.shape

            adversarials = utils.remove_failed(
                defended_model, images, labels, adversarials, True)

            for i in range(len(images)):
                # Move to CPU and save
                attack_results[i][test_name] = adversarials[i].cpu()

        images = images.cpu()
        labels = labels.cpu()

        all_images += list(images)
        all_true_labels += list(true_labels)
        all_attack_results += attack_results

    assert len(all_true_labels) == len(all_images)
    assert len(all_attack_results) == len(all_images)

    return adversarial_dataset.AttackComparisonDataset(all_images, all_true_labels, test_names, all_attack_results, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs)


def multiple_attack_test(model, attack_names, attacks, loader, p, misclassification_policy, device, attack_configuration, start, stop, generation_kwargs):
    assert all(not attack.targeted for attack in attacks)
    assert len(attack_names) == len(attacks)

    logger.debug('Running multiple attack tests with attacks %s.', attack_names)

    if misclassification_policy == 'remove':
        logger.warning('Remember that using "remove" as a misclassification policy can interfere with dataset merging.')

    model.to(device)

    all_images = []
    all_labels = []
    all_true_labels = []
    all_attack_results = []

    for images, true_labels in tqdm(loader, desc='Multiple Attack Test'):
        images = images.to(device)
        true_labels = true_labels.to(device)

        # true_labels are the labels in the dataset, while labels are
        # the labels that will be used by the attack (which may or may not
        # be the same as true_labels, depending on the misclassification policy)
        images, true_labels, labels = utils.apply_misclassification_policy(
            model, images, true_labels, misclassification_policy)

        if len(images) == 0:
            assert misclassification_policy == 'remove'
            logger.warning('0 images left after removing misclassified, skipping batch.')
            continue

        attack_results = [dict() for _ in range(len(images))]

        for test_name, attack in zip(attack_names, attacks):
            # Note y=labels
            adversarials = attack.perturb(images, y=labels).detach()
            assert adversarials.shape == images.shape

            adversarials = utils.remove_failed(
                model, images, labels, adversarials, False)

            for i in range(len(images)):
                # Move to CPU and save
                if adversarials[i] is None:
                    attack_results[i][test_name] = None
                else:
                    attack_results[i][test_name] = adversarials[i].cpu()

        images = images.cpu()
        labels = labels.cpu()
        true_labels = true_labels.cpu()

        all_images += list(images)
        all_labels += list(labels)
        all_true_labels += list(true_labels)
        all_attack_results += attack_results

    assert len(all_labels) == len(all_images)
    assert len(all_true_labels) == len(all_images)
    assert len(all_attack_results) == len(all_images)

    logger.debug('Collected %s results.', len(all_attack_results))

    return adversarial_dataset.AttackComparisonDataset(all_images, all_labels, all_true_labels, attack_names, all_attack_results, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs)
