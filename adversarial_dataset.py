import torch
import torch.utils.data as data

import utils

# TODO: Anche attack_configuration?
class AdversarialDataset(data.Dataset):
    def __init__(self, genuines, original_labels, adversarials, p, total_count, generation_kwargs):
        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(adversarials)

        self.genuines = genuines.detach().cpu()
        self.original_labels = original_labels.detach().cpu()
        self.adversarials = adversarials.detach().cpu()
        self.p = p
        self.total_count = total_count
        self.generation_kwargs = generation_kwargs

    @property
    def successful_count(self):
        return len(self.adversarials)

    @property
    def distances(self):
        return utils.adversarial_distance(self.genuines, self.adversarials, self.p)

    @property
    def attack_success_rate(self):
        return self.successful_count / self.total_count

    def to_distance_dataset(self):
        return AdversarialDistanceDataset(self.genuines, self.distances)

    def to_adversarial_training_dataset(self):
        return AdversarialTrainingDataset(self.adversarials, self.original_labels)

    def __getitem__(self, idx):
        return (self.genuines[idx], self.original_labels[idx], self.adversarials[idx])

    def __len__(self):
        return len(self.genuines)

class AdversarialTrainingDataset(data.Dataset):
    def __init__(self, adversarials, original_labels):
        assert len(adversarials) == len(original_labels)

        self.adversarials = adversarials.detach().cpu()
        self.original_labels = original_labels.detach().cpu()

    def __getitem__(self, idx):
        return (self.adversarials[idx], self.original_labels[idx])

    def __len__(self):
        return len(self.adversarials)

class AdversarialDistanceDataset(data.Dataset):
    def __init__(self, images, distances):
        assert len(images) == len(distances)

        self.images = images
        self.distances = distances

    def __getitem__(self, idx):
        return (self.images[idx], self.distances[idx].unsqueeze(0))

    def __len__(self):
        return len(self.images)

class EvasionDataset(data.Dataset):
    def __init__(self, genuines, original_labels, test_names, attack_results, p, generation_kwargs):
        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(attack_results)

        self.genuines = genuines.detach().cpu()
        self.original_labels = original_labels.detach().cpu()
        self.test_names = test_names
        self.attack_results = attack_results

        # Detach and convert to CPU each adversarial example
        for attack_result in self.attack_results:
            for key, value in attack_result.items():
                attack_result[key] = attack_result.detach().cpu()

        self.p = p
        self.generation_kwargs = generation_kwargs

    def __getitem__(self, idx):
        return (self.genuines[idx], self.original_labels[idx], self.attack_results[idx])

    def __len__(self):
        return len(self.genuines)

    def to_adversarial_dataset(self, test_name):
        genuines = []
        original_labels = []
        adversarials = []

        for genuine, original_label, attack_result in zip(self.genuines, self.original_labels, self.attack_results):
            if test_name in attack_result.keys():
                genuines.append(genuine)
                original_labels.append(original_label)
                adversarials.append(attack_result[test_name])

        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(adversarials)

        # torch.stack doesn't work with empty lists, so if necessary
        # we manually build empty tensors

        if len(genuines) > 0:
            genuines = torch.stack(genuines)
            original_labels = torch.stack(original_labels)
            adversarials = torch.stack(adversarials)
        else:
            genuines = torch.zeros((0,) + self.genuines.shape[1:])
            original_labels = torch.zeros((0,) + self.original_labels.shape[1:])
            adversarials = torch.zeros((0,) + self.genuines.shape[1:])

        
        return AdversarialDataset(genuines, original_labels, adversarials, self.p, len(self.genuines), self.generation_kwargs)

