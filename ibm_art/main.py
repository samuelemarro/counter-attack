import art
import tensorflow as tf
import tensorflow_datasets as datasets
import numpy as np
import torch
import cifar10_models
import matplotlib.pyplot as plt

import ignite

import adversarial_dataset
import utils
import approximator
import counter_attack
import detector_estimation

def get_model_and_attack(model, attack_name):
    # Pensavo di fare una cosa del genere:
    # - Controlla se l'attacco ha supporto native
    # - Se lo supporta, restituisce il modello native e l'attacco native
    # - Altrimenti restituisce modello e attacco standard
    pass

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
    
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    stdevs = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))

    loss = torch.nn.CrossEntropyLoss()
    optimizer = None

    model = art.classifiers.PyTorchClassifier(model, loss, optimizer, input_shape=[3, 32, 32],
                                                nb_classes=10,
                                                channel_index=1,
                                                clip_values=[0, 1],
                                                preprocessing=(mean, stdevs))

    #attack = art.attacks.CarliniL2Method(model, batch_size=100, binary_search_steps=5)
    attack = art.attacks.DeepFool(model, batch_size=100)

    # Ricorda che hai modificato il PyTorchModel! (con reduction='sum')

    # Ricorda che hai modificato art.DetectorModel! (per evitare length=0)

    ds_train = ds_train.shuffle(1000).batch(1).prefetch(10)

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

    detector_estimator = detector_estimation.CoordinateWiseDetectorEstimator(0.0001)

    detector = counter_attack.CounterAttackDetector(model, attack, p, 0.01, detector_estimator)

    detector_model = art.classifiers.DetectorClassifier(model, detector)

    #scores = detector.predict(dataset_adversarials)
    #gradients = detector.class_gradient(dataset_adversarials, label=0)

    images = image_batches[0]
    original_labels = label_batches[0]

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