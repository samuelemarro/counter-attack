import torch
from tqdm import tqdm

import detectors
import utils
import adversarial_dataset

def accuracy(model, loader, device):
    correct_count = 0
    total_count = 0

    model.to(device)

    for images, labels in tqdm(loader, desc='Accuracy Test'):
        total_count += len(images)
        images = images.to(device)
        labels = labels.to(device)

        predicted_labels = utils.get_labels(model, images)

        correct = torch.eq(predicted_labels, labels)
        correct_count += len(torch.nonzero(correct))

    return correct_count / total_count

def attack_test(model, attack, loader, p, remove_misclassified, device, generation_kwargs, attack_configuration, defended_model, blind_trust=False):
    assert not attack.targeted
    model.to(device)

    all_images = []
    all_labels = []
    all_adversarials = []

    for images, labels in tqdm(loader, desc='Attack Test'):
        images = images.to(device)
        labels = labels.to(device)

        image_shape = images.shape[1:]
        
        assert len(images) == len(labels)

        if remove_misclassified:
            images, labels = utils.remove_misclassified(model, images, labels)

        # TODO: Supporto politica misclassified?
        
        adversarials = attack.perturb(images, y=labels).detach()
        assert adversarials.shape == images.shape

        if blind_trust:
            adversarials = list(adversarials)
        else:
            if defended_model is None:
                adversarials = utils.remove_failed(model, images, labels, adversarials, False)
            else:
                adversarials = utils.remove_failed(defended_model, images, labels, adversarials, True)

        # Move to CPU
        images = images.cpu()
        labels = labels.cpu()
        for i in range(len(adversarials)):
            if adversarials[i] is not None:
                adversarials[i] = adversarials[i].cpu()

        all_images += list(images)
        all_labels += list(labels)
        all_adversarials += list(adversarials)

    
    assert len(all_images) == len(all_labels)
    assert len(all_images) == len(all_adversarials)

    return adversarial_dataset.AdversarialDataset(all_images, all_labels, all_adversarials, p, attack_configuration, generation_kwargs)

def mip_test(model, attack, loader, p, remove_misclassified, device, generation_kwargs, attack_configuration, pre_adversarial_dataset=None):
    assert not attack.targeted
    model.to(device)

    all_images = []
    all_labels = []
    all_adversarials = []
    all_lower_bounds = []
    all_upper_bounds = []
    all_solve_times = []

    for images, labels in tqdm(loader, desc='MIP Test'):
        images = images.to(device)
        labels = labels.to(device)

        image_shape = images.shape[1:]
        
        assert len(images) == len(labels)

        # TODO: La politica con i misclassified potrebbe scontrarsi con i controlli per i pre

        if remove_misclassified:
            images, labels = utils.remove_misclassified(model, images, labels)

        if pre_adversarial_dataset is None:
            pre_images = None
            pre_adversarials = None
        else:
            offset = len(all_images)
            count = len(images)
            pre_images = pre_adversarial_dataset.genuines[offset:offset+count]
            pre_adversarials = pre_adversarial_dataset.adversarials[offset:offset+count]
            for i in range(len(pre_images)):
                pre_images[i] = pre_images[i].to(device)
                if pre_adversarials[i] is not None:
                    pre_adversarials[i] = pre_adversarials[i].to(device)
        
        # TODO: Supporto politica per misclassified

        if pre_adversarial_dataset is not None:
            # Check that the images are the same
            all_match = all([torch.max(torch.abs(image - pre_image)) < 1e-6 for image, pre_image in zip(images, pre_images)])

            if not all_match:
                raise RuntimeError('The pre-adversarials refer to different genuines. '
                                   'This can slow down MIP at best and make it fail at worst. '
                                   'Are you sure that you\'re using the correct pre-adversarial dataset?')
        adversarials, lower_bounds, upper_bounds, solve_times = attack.perturb_advanced(images, y=labels, starting_points=pre_adversarials)
        assert len(adversarials) == len(images)
        assert len(adversarials) == len(lower_bounds)
        assert len(adversarials) == len(upper_bounds)
        assert len(adversarials) == len(solve_times)

        # Move to CPU
        images = images.cpu()
        labels = labels.cpu()

        all_images += list(images)
        all_labels += list(labels)
        all_adversarials += list(adversarials)
        all_lower_bounds += list(lower_bounds)
        all_upper_bounds += list(upper_bounds)
        all_solve_times += list(solve_times)
    
    assert len(all_images) == len(all_labels)
    assert len(all_images) == len(all_adversarials)
    assert len(all_images) == len(all_lower_bounds)
    assert len(all_images) == len(all_upper_bounds)
    assert len(all_images) == len(all_solve_times)

    return adversarial_dataset.MIPDataset(all_images, all_labels, all_adversarials, all_lower_bounds, all_upper_bounds, all_solve_times, p, attack_configuration, generation_kwargs)

def multiple_evasion_test(model, test_names, attacks, defended_models, loader, p, remove_misclassified, device, attack_configuration, generation_kwargs):
    assert all(not attack.targeted for attack in attacks)
    assert all(attack.predict == defended_model.predict for attack, defended_model in zip(attacks, defended_models))

    model.to(device)

    for defended_model in defended_models:
        defended_model.to(device)

    assert len(test_names) == len(attacks)
    assert len(test_names) == len(defended_models)

    all_images = []
    all_labels = []
    all_attack_results = []

    total_count = 0

    for images, labels in tqdm(loader, desc='Multiple Evasion Test'):
        total_count += len(images)

        images = images.to(device)
        labels = labels.to(device)

        image_shape = images.shape[1:]
        
        if remove_misclassified:
            images, labels = utils.remove_misclassified(model, images, labels)

        attack_results = [dict() for _ in range(len(images))]

        for test_name, attack, defended_model in zip(test_names, attacks, defended_models):
            # Nota y=labels
            adversarials = attack.perturb(images, y=labels).detach()

            assert adversarials.shape == images.shape

            adversarials = utils.remove_failed(defended_model, images, labels, adversarials, True)

            for i in range(len(images)):
                # Move to CPU and save
                attack_results[i][test_name] = adversarials[i].cpu()
            
        images = images.cpu()
        labels = labels.cpu()

        all_images += list(images)
        all_labels += list(labels)
        all_attack_results += attack_results

    assert len(all_labels) == len(all_images)
    assert len(all_attack_results) == len(all_images)

    return adversarial_dataset.AttackComparisonDataset(all_images, all_labels, test_names, all_attack_results, p, attack_configuration, generation_kwargs)
        
def multiple_attack_test(model, attack_names, attacks, loader, p, remove_misclassified, device, attack_configuration, generation_kwargs):
    assert all(not attack.targeted for attack in attacks)

    model.to(device)

    assert len(attack_names) == len(attacks)

    all_images = []
    all_labels = []
    all_attack_results = []

    total_count = 0

    for images, labels in tqdm(loader, desc='Multiple Attack Test'):
        total_count += len(images)

        images = images.to(device)
        labels = labels.to(device)

        image_shape = images.shape[1:]
        
        if remove_misclassified:
            images, labels = utils.remove_misclassified(model, images, labels)

        attack_results = [dict() for _ in range(len(images))]

        for test_name, attack in zip(attack_names, attacks):
            # Nota y=labels
            adversarials = attack.perturb(images, y=labels).detach()

            assert adversarials.shape == images.shape
            
            adversarials = utils.remove_failed(model, images, labels, adversarials, False)

            for i in range(len(images)):
                # Move to CPU and save
                if adversarials[i] is None:
                    attack_results[i][test_name] = None
                else:
                    attack_results[i][test_name] = adversarials[i].cpu()

        images = images.cpu()
        labels = labels.cpu()

        all_images += list(images)
        all_labels += list(labels)
        all_attack_results += attack_results

    assert len(all_labels) == len(all_images)
    assert len(all_attack_results) == len(all_images)

    return adversarial_dataset.AttackComparisonDataset(all_images, all_labels, attack_names, all_attack_results, p, attack_configuration, generation_kwargs)