import art
import tensorflow as tf
import tensorflow_datasets as datasets
import numpy as np
import torch
import cifar10_models
import matplotlib.pyplot as plt
import torchvision

import ignite

import adversarial_dataset
import utils
import approximator

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

    attack = art.attacks.CarliniL2Method(model, batch_size=100, binary_search_steps=5)
    #attack = foolbox.attacks.DeepFoolAttack(model=model, distance=foolbox.distances.Linfinity)
    import foolbox
    #attack = foolbox.attacks.CarliniWagnerL2Attack(model=model)

    # Ricorda che hai modificato il PyTorchModel! (con reduction='sum')

    ds_train = ds_train.shuffle(1000).batch(100).prefetch(10)
    image_count = 0
    correct_count = 0
    adversarial_count = 0

    dataset_adversarials = []
    dataset_distances = []


    for features in ds_train.take(3):
        images, labels = features["image"], features["label"]
        images = images.numpy().astype(np.float32)
        labels = labels.numpy().astype(np.int32)

        images = np.moveaxis(images, 3, 1)
        images /= 255

        image_count += len(images)

        images, labels = remove_misclassified(model, images, labels)

        correct_count += len(images)

        adversarials = attack.generate(images, labels)
        images, labels, adversarials = remove_failed(images, labels, adversarials)

        adversarial_count += len(images)

        print('{} samples, {} correct, {} adversarial (success rate: {:.2f}%)'.format(image_count, correct_count, adversarial_count, adversarial_count / correct_count * 100.0))

        # L-infinity
        distances = np.max(np.abs(images - adversarials), axis=(1, 2, 3))
        print(distances)


        #images = np.moveaxis(images, 1, 3)
        #adversarials = np.moveaxis(adversarials, 1, 3)


        dataset_adversarials += list(adversarials)
        dataset_distances += list(distances)

        """f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(images[0])
        axarr[0,1].imshow(adversarials[0])
        axarr[1,0].imshow(images[1])
        axarr[1,1].imshow(adversarials[1])
        plt.show()"""

    dataset_adversarials = np.array(dataset_adversarials)
    dataset_distances = np.array(dataset_distances)

    dataset_distances = np.expand_dims(dataset_distances, -1)

    approximator_model = approximator.get_approximator()

    approximator.train_approximator(approximator_model, dataset_adversarials, dataset_distances, 1000, 1e-4, 0)

    adversarial_distance_dataset = adversarial_dataset.AdversarialDistanceDataset('deepfool', dataset_adversarials, dataset_distances)
    utils.save_zip(adversarial_distance_dataset, 'deepfool_cifar10_resnet50.zip')
        

if __name__ == '__main__':
    main()