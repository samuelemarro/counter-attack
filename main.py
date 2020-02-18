import art
#import tensorflow as tf
import tensorflow_datasets as datasets
import numpy as np
import torch
import cifar10_models
import matplotlib.pyplot as plt

import ignite

import advertorch

import adversarial_dataset
import utils
import torch_estimation
import torch_utils
import detectors

import custom_attacks
import sanity_tests

def get_model_and_attack(model, attack_name):
    # Pensavo di fare una cosa del genere:
    # - Controlla se l'attacco ha supporto native
    # - Se lo supporta, restituisce il modello native e l'attacco native
    # - Altrimenti restituisce modello e attacco standard
    pass

# Togliere?
def remove_misclassified(model, images, labels):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=-1)
    
    correct_label = np.equal(predicted_labels, labels)

    return images[correct_label], labels[correct_label]

def remove_failed(images, labels, adversarials):
    successful = np.array([x is not None for x in adversarials])
    successful_adversarials = np.array([x for x in adversarials if x is not None])
    return images[successful], labels[successful], successful_adversarials

def get_adversarials(model, attack, iterator, p):
    image_count = 0
    correct_count = 0
    adversarial_count = 0

    dataset_adversarials = []
    dataset_distances = []
    
    for images, labels in iterator:
        image_count += len(images)

        images, labels = remove_misclassified(model, images, labels)

        correct_count += len(images)

        adversarials = attack.generate(images, labels)
        images, labels, adversarials = remove_failed(images, labels, adversarials)

        adversarial_count += len(images)

        print('{} samples, {} correct, {} adversarial (success rate: {:.2f}%)'.format(image_count, correct_count, adversarial_count, adversarial_count / correct_count * 100.0))

        differences = images - adversarials
        #Controllare
        differences = differences.reshape([differences.shape[0], -1])

        distances = np.linalg.norm(differences, ord=p, axis=1)
        print(distances)

        dataset_adversarials += list(adversarials)
        dataset_distances += list(distances)

        """
        images = np.moveaxis(images, 1, 3)
        adversarials = np.moveaxis(adversarials, 1, 3)

        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(images[0])
        axarr[0,1].imshow(adversarials[0])
        axarr[1,0].imshow(images[1])
        axarr[1,1].imshow(adversarials[1])
        plt.show()"""

    dataset_adversarials = np.array(dataset_adversarials)
    dataset_distances = np.array(dataset_distances)

    return dataset_adversarials, dataset_distances



def main():
    
    ds_train = datasets.load(name="cifar10", split="train", shuffle_files=True)

    model = cifar10_models.resnet50(pretrained=True)
    model.eval()
    
    mean = np.array([0.4914, 0.4822, 0.4465])#.reshape((3, 1, 1))
    stdevs = np.array([0.2023, 0.1994, 0.2010])#.reshape((3, 1, 1))

    normalisation = torch_utils.Normalisation(mean, stdevs)

    model = torch.nn.Sequential(normalisation, model)

    # Ricorda che hai modificato il foolbox.PyTorchModel! (con reduction='sum')

    # Ricorda che hai modificato art.DetectorModel! (per evitare length=0)

    ds_train = ds_train.shuffle(1000).batch(50).prefetch(10)

    image_batches = []
    label_batches = []
    
    for features in ds_train.take(1):
        images, labels = features["image"], features["label"]
        images = images.numpy().astype(np.float32)
        labels = labels.numpy().astype(np.int32)
        images = np.moveaxis(images, 3, 1)
        images /= 255

        image_batches.append(images)
        label_batches.append(labels)

    iterator = zip(image_batches, label_batches)

    p = np.inf

    #dataset_adversarials, dataset_distances = get_adversarials(model, attack, iterator, p)

    #detector_estimator = detector_estimation.CoordinateWiseDetectorEstimator(0.0001)

    #detector = counter_attack.CounterAttackDetector(model, attack, p, 0.01, detector_estimator)

    #detector_model = art.classifiers.DetectorClassifier(model, detector)

    #scores = detector.predict(dataset_adversarials)
    #gradients = detector.class_gradient(dataset_adversarials, label=0)

    device = 'cuda'

    images = image_batches[0]
    original_labels = label_batches[0]

    images = torch.from_numpy(images).to(device).float()
    
    original_labels = torch.from_numpy(original_labels).to(device)
    model.to(device)

    model = torch_utils.BatchLimitedModel(model, 200).float()
    
    #test_gradient_estimation(model, torch_estimation.BinomialSamplingEstimator_F, images, 1e-5, 1000)


    #attack = advertorch.attacks.CarliniWagnerL2Attack(model.forward, 10, max_iterations=1000)
    attack_1 = advertorch.attacks.GradientSignAttack(model)
    attack_2 = advertorch.attacks.FGSM(model)
    attack_3 = advertorch.attacks.CarliniWagnerL2Attack(model, 10, max_iterations=100)
    attack_pool = [attack_1, attack_2, attack_3]

    #attack = custom_attacks.AttackPool(attack_pool, 2)
    #attack = custom_attacks.RandomTargetAttackAgainstDetector(model, attack_1)
    #attack = custom_attacks.KBestTargetAttackAgainstDetector(model, attack_1)
    attack = attack_1

    #attack.perturb(images)

    
    detector = detectors.CounterAttackDetector(attack, 2)

    #detector_estimator = torch_estimation.CoordinateWiseEstimator(detector, 1e-3)

    # TODO: Controllare che la normalizzazione sia corretta
    #detector_model = detectors.NormalisedDetectorModel(model, detector_estimator, -16)

    #final_outputs = detector_model(images)
    #print(final_outputs)

    #sanity_tests.test_detector_output(model, detector, -5, images)
    #sanity_tests.test_gradient_estimation(model, torch_estimation.CoordinateWiseEstimator_F, images, 1e-3)
    #sanity_tests.test_model_batch_limiting(model, images, 100)
    #sanity_tests.test_normalisation(model, images)

    detector_scores = detector(images)

    import copy

    distance_dataset = adversarial_dataset.AdversarialDistanceDataset('fgsm', images.detach(), detector_scores.detach())
    distance_loader = torch.utils.data.DataLoader(distance_dataset, batch_size=50, shuffle=True)
    val_dataset = adversarial_dataset.AdversarialDistanceDataset('fgsm', images.clone(), detector_scores.clone())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=True)

    approximator_model = approximator.get_approximator()
    torch_utils.train(approximator_model, distance_loader, val_loader, torch.optim.SGD(approximator_model.parameters(), 1e-5), torch.nn.MSELoss(), 100)

    return

    target_labels = []

    for label in original_labels:
        target_label = np.random.randint(10)
        while target_label == label.item():
            target_label = np.random.randint(10)

        target_labels.append(target_label)

    target_labels = np.array(target_labels)

    print(images.shape)
    print(target_labels.shape)

    double_attack = art.attacks.CarliniL2Method(detector_model, targeted=True)

    image_count = len(images)

    adversarials = double_attack.generate(images, y=target_labels)

    adversarial_count = len(adversarials)

    differences = images - adversarials
    #Controllare
    differences = differences.reshape([differences.shape[0], -1])

    distances = np.linalg.norm(differences, ord=p, axis=1)
    print('Success Rate: {:.2f}%'.format(adversarial_count / image_count * 100.0))
    print(distances)

    #print(gradients)

    #dataset_distances = np.expand_dims(dataset_distances, -1)

    approximator_model = approximator.get_approximator()

    approximator.train_approximator(approximator_model, dataset_adversarials, dataset_distances, 1000, 1e-4, 0)

    adversarial_distance_dataset = adversarial_dataset.AdversarialDistanceDataset('deepfool', dataset_adversarials, dataset_distances)
    utils.save_zip(adversarial_distance_dataset, 'deepfool_cifar10_resnet50.zip')
    
if __name__ == '__main__':
    main()