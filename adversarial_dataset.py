import collections

import numpy as np
import torch
import torch.utils.data as data

import utils

class AdversarialDataset(data.Dataset):
    def __init__(self, genuines, original_labels, adversarials, p, total_count, attack_configuration, generation_kwargs):
        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(adversarials)
        assert len(genuines) <= total_count

        self.genuines = genuines.detach().cpu()
        self.original_labels = original_labels.detach().cpu()
        self.adversarials = adversarials.detach().cpu()
        self.p = p
        self.total_count = total_count
        self.attack_configuration = attack_configuration
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

    def print_stats(self):
        distances = self.distances.numpy()

        success_rate = self.attack_success_rate
        median_distance = np.median(distances)
        average_distance = np.average(distances)

        print('Success Rate: {:.2f}%'.format(success_rate * 100.0))
        print('Median Distance: {}'.format(median_distance))
        print('Average Distance: {}'.format(average_distance))

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

class EvasionResultDataset(data.Dataset):
    def __init__(self, genuines, original_labels, test_names, attack_results, p, attack_configuration, generation_kwargs):
        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(attack_results)

        self.genuines = genuines.detach().cpu()
        self.original_labels = original_labels.detach().cpu()
        self.test_names = test_names
        self.attack_results = attack_results

        # Detach and convert to CPU each adversarial example
        for attack_result in self.attack_results:
            for key, value in attack_result.items():
                attack_result[key] = value.detach().cpu()

        self.p = p
        self.attack_configuration = attack_configuration
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

        # torch.stack doesn't work with empty lists, so in such cases we
        # return a tensor with 0 as the first dimension

        genuines = utils.maybe_stack(genuines, self.genuines.shape[1:])
        original_labels = utils.maybe_stack(original_labels, None, torch.long)
        adversarials = utils.maybe_stack(adversarials, self.genuines.shape[1:])
        
        return AdversarialDataset(genuines, original_labels, adversarials, self.p, len(self.genuines), self.attack_configuration, self.generation_kwargs)

class AttackComparisonDataset(data.Dataset):
    def __init__(self, genuines, original_labels, attack_names, attack_results, p, attack_configuration, generation_kwargs):
        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(attack_results)

        self.genuines = genuines.detach().cpu()
        self.original_labels = original_labels.detach().cpu()
        self.attack_names = attack_names
        self.attack_results = attack_results

        # Detach and convert to CPU each adversarial example
        for attack_result in self.attack_results:
            for key, value in attack_result.items():
                attack_result[key] = value.detach().cpu()

        self.p = p
        self.attack_configuration = attack_configuration
        self.generation_kwargs = generation_kwargs

    def __getitem__(self, idx):
        return (self.genuines[idx], self.original_labels[idx], self.attack_results[idx])

    def __len__(self):
        return len(self.genuines)

    def to_adversarial_dataset(self, attack_names):
        genuines = []
        original_labels = []
        adversarials = []

        for genuine, original_label, attack_result in zip(self.genuines, self.original_labels, self.attack_results):
            if attack_names in attack_result.keys():
                genuines.append(genuine)
                original_labels.append(original_label)
                adversarials.append(attack_result[attack_names])

        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(adversarials)

        # torch.stack doesn't work with empty lists, so in such cases we
        # return a tensor with 0 as the first dimension

        genuines = utils.maybe_stack(genuines, self.genuines.shape[1:])
        original_labels = utils.maybe_stack(original_labels, None, torch.long)
        adversarials = utils.maybe_stack(adversarials, self.genuines.shape[1:])
        
        return AdversarialDataset(genuines, original_labels, adversarials, self.p, len(self.genuines), self.attack_configuration, self.generation_kwargs)

    def simulate_pooling(self, selected_attacks):
        genuines = []
        original_labels = []
        best_adversarials = []

        for genuine, original_label, attack_result in zip(self.genuines, self.original_labels, self.attack_results):
            # Take the adversarials generated by attacks that were successful and
            # were selected by the user
            chosen_adversarials = [attack_result[attack_names] for attack_names in selected_attacks if attack_names in attack_result]

            if len(chosen_adversarials) > 0:
                genuines.append(genuine)
                original_labels.append(original_label)

                chosen_adversarials = torch.stack(chosen_adversarials)
                distances = utils.one_many_adversarial_distance(genuine, chosen_adversarials, self.p)

                assert distances.shape == (len(chosen_adversarials),)

                best_distance_index = torch.argmin(distances)
            
                best_adversarials.append(chosen_adversarials[best_distance_index])

        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(best_adversarials)

        # torch.stack doesn't work with empty lists, so in such cases we
        # return a tensor with 0 as the first dimension

        genuines = utils.maybe_stack(genuines, self.genuines.shape[1:])
        original_labels = utils.maybe_stack(original_labels, None, torch.long)
        best_adversarials = utils.maybe_stack(best_adversarials, self.genuines.shape[1:])

        return AdversarialDataset(genuines, original_labels, best_adversarials, self.p, len(self.genuines), self.attack_configuration, self.generation_kwargs)

    def attack_ranking_stats(self, attack_name):
        attack_positions = dict()
        for i in range(len(self.attack_names)):
            attack_positions[i] = 0
        
        attack_positions['failure'] = 0

        for genuine, original_label, attack_result in zip(self.genuines, self.original_labels, self.attack_results):
            if attack_name in attack_result:
                # Note: dictionaries don't preserve order, so we convert to OrderedDict
                attack_result = collections.OrderedDict(attack_result)
                
                adversarials = torch.stack(list(attack_result.values()))
                distances = utils.one_many_adversarial_distance(genuine, adversarials, self.p)

                assert len(attack_result) == len(distances)

                attack_distance_pairs = list(zip(attack_result.keys(), distances))

                # Sort by distance in ascending order (lower distance = better)
                attack_distance_pairs = sorted(attack_distance_pairs, key=lambda x: x[1])
                sorted_test_names = [x[0] for x in attack_distance_pairs]

                attack_ranking = sorted_test_names.index(attack_name)

                attack_positions[attack_ranking] += 1
            else:
                attack_positions['failure'] += 1

        assert sum(count for count in attack_positions.values()) == len(self.genuines)

        # Convert absolute numbers to relative
        for key in attack_positions.keys():
            attack_positions[key] /= len(self.genuines)

        return attack_positions
    
    def pairwise_comparison(self):
        victory_matrix = dict()

        # Initialize the matrix
        for attack_name in self.attack_names:
            victory_matrix[attack_name] = dict()
            for other_attack_name in [x for x in self.attack_names if x != attack_name]:
                victory_matrix[attack_name][other_attack_name] = 0

        for genuine, original_label, attack_result in zip(self.genuines, self.original_labels, self.attack_results):
            # Note: dictionaries don't preserve order, so we convert to OrderedDict
            attack_result = collections.OrderedDict(attack_result)

            successful_attacks = attack_result.keys()
            unsuccessful_attacks = [x for x in self.attack_names if x not in successful_attacks]

            # Successful attacks always beat unsuccessful attacks
            for successful_attack in successful_attacks:
                for unsuccessful_attack in unsuccessful_attacks:
                    victory_matrix[successful_attack][unsuccessful_attack] += 1

            if len(attack_result.values()) > 0:
                adversarials = torch.stack(list(attack_result.values()))
                distances = utils.one_many_adversarial_distance(genuine, adversarials, self.p)

                assert len(attack_result) == len(distances)

                attack_distance_pairs = list(zip(attack_result.keys(), distances))

                for winner_attack, winner_distance in attack_distance_pairs:
                    for loser_attack, loser_distance in attack_distance_pairs:
                        # An attack beats another if it finds a strictly smaller
                        # distance
                        if winner_distance < loser_distance:
                            victory_matrix[winner_attack][loser_attack] += 1

        # Convert absolute numbers to relative
        for loser_dict in victory_matrix.values():
            for key2 in loser_dict.keys():
                loser_dict[key2] /= len(self.genuines)

        return victory_matrix