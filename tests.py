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

        predicted_labels = torch.argmax(model(images), dim=1)
        assert predicted_labels.shape == labels.shape

        correct = torch.eq(predicted_labels, labels)
        correct_count += len(torch.nonzero(correct))

    return correct_count / total_count

def attack_test(model, attack, loader, p, remove_misclassified, device, generation_kwargs, attack_configuration, defended_model):
    model.to(device)

    total_count = 0

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

        total_count += len(images)
        
        adversarials = attack.perturb(images, y=labels)
        assert adversarials.shape == images.shape

        if defended_model is None:
            images, labels, adversarials = utils.remove_failed(model, images, labels, adversarials, False)
        else:
            images, labels, adversarials = utils.remove_failed(defended_model, images, labels, adversarials, True)

        all_images += list(images)
        all_labels += list(labels)
        all_adversarials += list(adversarials)

    # torch.stack doesn't work with empty lists, so in such cases we return a tensor with 0 as the first dimension
    all_images = utils.maybe_stack(all_images, image_shape).to(device)
    all_labels = utils.maybe_stack(all_labels, None, dtype=torch.long).to(device)
    all_adversarials = utils.maybe_stack(all_adversarials, image_shape).to(device)

    assert all_images.shape == all_adversarials.shape
    assert len(all_images) == len(all_labels)

    return adversarial_dataset.AdversarialDataset(all_images, all_labels, all_adversarials, p, total_count, attack_configuration, generation_kwargs)

def multiple_evasion_test(model, test_names, attacks, defended_models, loader, p, remove_misclassified, device, attack_configuration, generation_kwargs):
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
            adversarials = attack.perturb(images, y=labels)

            assert adversarials.shape == images.shape
            
            successful = utils.check_success(defended_model, adversarials, labels, True)

            for i in range(len(images)):
                if successful[i]:
                    attack_results[i][test_name] = adversarials[i]

        all_images += list(images)
        all_labels += list(labels)
        all_attack_results += attack_results

    # torch.stack doesn't work with empty lists, so in such cases we
    # return a tensor with 0 as the first dimension

    all_images = utils.maybe_stack(all_images, image_shape).to(device)
    all_labels = utils.maybe_stack(all_labels, None, dtype=torch.long).to(device)

    assert len(all_labels) == len(all_images)
    assert len(all_attack_results) == len(all_images)

    return adversarial_dataset.EvasionDataset(all_images, all_labels, test_names, attack_results, p, attack_configuration, generation_kwargs)

        
    