import collections

import numpy as np
import torch
import torch.utils.data as data

import utils

class AdversarialDataset(data.Dataset):
    def __init__(self, genuines, original_labels, adversarials, p, attack_configuration, generation_kwargs):
        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(adversarials)

        self.genuines = [genuine.detach().cpu() for genuine in genuines]
        self.original_labels = [original_label.detach().cpu() for original_label in original_labels]
        self.adversarials = []
        for adversarial in adversarials:
            if adversarial is None:
                self.adversarials.append(None)
            else:
                self.adversarials.append(adversarial.detach().cpu())
        self.p = p
        self.attack_configuration = attack_configuration
        self.generation_kwargs = generation_kwargs

    @property
    def successful_count(self):
        return len([x for x in self.adversarials if x is not None])

    @property
    def successful_distances(self):
        successful_genuines = []
        successful_adversarials = []

        for genuine, adversarial in zip(self.genuines, self.adversarials):
            if adversarial is not None:
                successful_genuines.append(genuine)
                successful_adversarials.append(adversarial)

        if len(successful_genuines) > 0:
            successful_genuines = torch.stack(successful_genuines)
            successful_adversarials = torch.stack(successful_adversarials)
            return utils.adversarial_distance(successful_genuines, successful_adversarials, self.p)
        else:
            return torch.empty(0, dtype=torch.float32)

    @property
    def attack_success_rate(self):
        return self.successful_count / len(self.genuines)

    def to_distance_dataset(self, failure_value=None):
        successful_distances = list(self.successful_distances)
        final_distances = []
        for i in range(len(self.genuines)):
            if self.adversarials[i] is None:
                final_distances.append(failure_value)
            else:
                final_distances.append(successful_distances.pop(0))
        
        assert len(successful_distances) == 0

        return AdversarialDistanceDataset(self.genuines, final_distances)

    def successful_adversarials(self):
        successful_genuines = []
        successful_adversarials = []
        successful_original_labels = []

        for genuine, adversarial, original_label in zip(self.genuines, self.adversarials, self.original_labels):
            if adversarial is not None:
                successful_genuines.append(genuine)
                successful_adversarials.append(adversarial)
                successful_original_labels.append(original_label)

        return successful_genuines, successful_adversarials, successful_original_labels

    def to_adversarial_training_dataset(self):
        _, successful_adversarials, successful_original_labels = self.successful_adversarials()
        return AdversarialTrainingDataset(successful_adversarials, successful_original_labels)

    def __getitem__(self, idx):
        return (self.genuines[idx], self.original_labels[idx], self.adversarials[idx])

    def __len__(self):
        return len(self.genuines)

    def print_stats(self):
        distances = self.successful_distances.numpy()

        success_rate = self.attack_success_rate
        median_distance = np.median(distances)
        average_distance = np.average(distances)

        print('Success Rate: {:.2f}%'.format(success_rate * 100.0))
        print('Median Successful Distance: {}'.format(median_distance))
        print('Average Successful Distance: {}'.format(average_distance))

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
        return (self.images[idx], self.distances[idx])

    def __len__(self):
        return len(self.images)

# TODO: attack_names è troppo specifico, serve un nome più generale

class AttackComparisonDataset(data.Dataset):
    def __init__(self, genuines, original_labels, attack_names, attack_results, p, attack_configuration, generation_kwargs):
        assert len(genuines) == len(original_labels)
        assert len(genuines) == len(attack_results)

        self.genuines = [genuine.detach().cpu() for genuine in genuines]
        self.original_labels = [original_label.detach().cpu() for original_label in original_labels]
        self.attack_names = attack_names
        self.attack_results = attack_results

        # Detach and convert to CPU each adversarial example
        for attack_result in self.attack_results:
            for key, value in attack_result.items():
                if value is not None:
                    attack_result[key] = value.detach().cpu()

        self.p = p
        self.attack_configuration = attack_configuration
        self.generation_kwargs = generation_kwargs

    def __getitem__(self, idx):
        return (self.genuines[idx], self.original_labels[idx], self.attack_results[idx])

    def __len__(self):
        return len(self.genuines)

    def to_adversarial_dataset(self, attack_name):
        genuines = []
        original_labels = []
        adversarials = [attack_result[attack_name] for attack_result in self.attack_results]
        
        return AdversarialDataset(self.genuines, self.original_labels, adversarials, self.p, len(self.genuines), self.attack_configuration, self.generation_kwargs)

    def simulate_pooling(self, selected_attacks):
        best_adversarials = []

        for genuine, original_label, attack_result in zip(self.genuines, self.original_labels, self.attack_results):
            # Take the adversarials generated by attacks that were successful and
            # were selected by the user
            successful_adversarials = [attack_result[attack_name] for attack_name in selected_attacks if attack_result[attack_name] is not None]

            if len(successful_adversarials) > 0:
                successful_adversarials = torch.stack(successful_adversarials)
                distances = utils.one_many_adversarial_distance(genuine, successful_adversarials, self.p)

                assert distances.shape == (len(successful_adversarials),)

                best_distance_index = torch.argmin(distances)
            
                best_adversarials.append(successful_adversarials[best_distance_index])
            else:
                best_adversarials.append(None)

        assert len(self.genuines) == len(best_adversarials)

        return AdversarialDataset(self.genuines, self.original_labels, best_adversarials, self.p, self.attack_configuration, self.generation_kwargs)

    def attack_ranking_stats(self, attack_name):
        attack_positions = dict()
        for i in range(len(self.attack_names)):
            attack_positions[i] = 0
        
        attack_positions['failure'] = 0

        for genuine, original_label, attack_result in zip(self.genuines, self.original_labels, self.attack_results):
            if attack_result[attack_name] is None:
                attack_positions['failure'] += 1
            else:
                attack_result = [(image, adversarial) for image, adversarial in attack_result.items() if adversarial is not None]
                # Note: dictionaries don't preserve order, so we convert to OrderedDict
                attack_result = collections.OrderedDict(attack_result)
                
                adversarials = torch.stack(list(attack_result.values()))
                distances = utils.one_many_adversarial_distance(genuine, adversarials, self.p)

                assert len(attack_result) == len(distances)

                attack_distance_pairs = list(zip(attack_result.keys(), distances))

                # Sort by distance in ascending order (lower distance = better)
                attack_distance_pairs = sorted(attack_distance_pairs, key=lambda x: x[1])
                sorted_test_names = [x[0] for x in attack_distance_pairs]

                # TODO: Come gestire uguali?

                attack_ranking = sorted_test_names.index(attack_name)

                attack_positions[attack_ranking] += 1

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

            successful_attacks = [name for name in self.attack_names if attack_result[name] is not None]
            unsuccessful_attacks = [name for name in self.attack_names if attack_result[name] is None]

            # Successful attacks always beat unsuccessful attacks
            for successful_attack in successful_attacks:
                for unsuccessful_attack in unsuccessful_attacks:
                    victory_matrix[successful_attack][unsuccessful_attack] += 1

            if len(attack_result.values()) > 0:
                adversarials = [x for x in attack_result.values() if x is not None]
                adversarials = torch.stack(adversarials)
                distances = utils.one_many_adversarial_distance(genuine, adversarials, self.p)

                assert len(successful_attacks) == len(distances)

                attack_distance_pairs = list(zip(successful_attacks, distances))

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