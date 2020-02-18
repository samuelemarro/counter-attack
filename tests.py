import torch
from tqdm import tqdm

import detectors
import utils
import adversarial_dataset

"""def counter_attack_matrix(model, images, labels, inner_attack_configurations, outer_attack_configurations, p, threshold, detector_estimator):
    evasion_results = []
    original_count = len(images)
    images, labels = utils.remove_misclassified(model, images, labels)
    correct_count = len(images)

    for inner_attack_configuration in inner_attack_configurations:
        inner_attack = inner_attack_configuration.generator(model)

        detector = detectors.CounterAttackDetector(inner_attack, p)
        detector_model = detectors.NormalisedDetectorModel(model, detector, threshold)

        for outer_attack_configuration in outer_attack_configurations:
            outer_attack = outer_attack_configuration.generator(detector_model)

            adversarials = outer_attack.generate(images)

            evasion_result = EvasionResult(inner_attack_configuration, outer_attack_configuration, images, adversarials, original_count, correct_count, p, counter_attack_threshold)

            evasion_results.append(evasion_result)


    return evasion_results"""

# Qual è la distribuzione degli score su immagini che ingannano il detector?
def transfer_test(target_model, detector):
    pass

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

# TODO: Passare anche l'attack configuration?
# TODO: has_detector non deve essere opzionale
def attack_test(model, attack, loader, p, remove_misclassified, device, generation_kwargs, has_detector):
    model.to(device)

    total_count = 0

    all_images = []
    all_labels = []
    all_adversarials = []

    for images, labels in tqdm(loader, desc='Attack Test'):
        images = images.to(device)
        labels = labels.to(device)
        
        assert len(images) == len(labels)
        #print(images.shape)

        if remove_misclassified:
            images, labels = utils.remove_misclassified(model, images, labels)

        total_count += len(images)
        
        #print(images.shape)
        # Nota y=labels
        adversarials = attack.perturb(images, y=labels)
        assert adversarials.shape == images.shape

        # TODO: Fare remove_failed il default?
        images, labels, adversarials = utils.remove_failed(model, images, labels, adversarials, has_detector)

        all_images += list(images)
        all_labels += list(labels)
        all_adversarials += list(adversarials)

    all_images = torch.stack(all_images)
    all_labels = torch.stack(all_labels)
    all_adversarials = torch.stack(all_adversarials)

    return adversarial_dataset.AdversarialDataset(all_images, all_labels, all_adversarials, p, total_count, generation_kwargs)

def multiple_evasion_test(model, test_names, attacks, defended_models, loader, p, remove_misclassified, device, generation_kwargs, has_detector):
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
        
        if remove_misclassified:
            images, labels = utils.remove_misclassified(model, images, labels)

        attack_results = [dict() for _ in range(len(images))]

        for test_name, attack, defended_model in zip(test_names, attacks, defended_models):
            # Nota y=labels
            adversarials = attack.perturb(images, y=labels)

            assert adversarials.shape == images.shape
            
            successful = utils.check_success(defended_model, images, labels, adversarials, has_detector)

            for i in range(len(images)):
                if successful[i]:
                    attack_results[i][test_name] = adversarials[i]

        all_images += list(images)
        all_labels += list(labels)
        all_attack_results += attack_results
        break # TODO: Rimuovere

    all_images = torch.stack(all_images)
    all_labels = torch.stack(all_labels)

    return adversarial_dataset.EvasionDataset(all_images, all_labels, test_names, attack_results, p, total_count)

        
    