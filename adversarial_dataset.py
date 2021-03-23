import collections

import numpy as np
import torch
import torch.utils.data as data

import utils

class AdversarialDataset(data.Dataset):
    def __init__(self, genuines, labels, true_labels, adversarials, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs):
        assert len(genuines) == len(labels)
        assert len(genuines) == len(true_labels)
        assert len(genuines) == len(adversarials)

        self.genuines = [genuine.detach().cpu() for genuine in genuines]
        self.labels = [label.detach().cpu() for label in labels]
        self.true_labels = [true_label.detach().cpu()
                            for true_label in true_labels]

        self.adversarials = []
        for adversarial in adversarials:
            if adversarial is None:
                self.adversarials.append(None)
            else:
                self.adversarials.append(adversarial.detach().cpu())

        self.p = p
        self.misclassification_policy = misclassification_policy
        self.attack_configuration = attack_configuration
        self.start = start
        self.stop = stop
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

        # Check that all successful distances were matched to a successful adversarial
        assert len(successful_distances) == 0

        return AdversarialDistanceDataset(self.genuines, final_distances)

    def successful_adversarials(self):
        successful_genuines = []
        successful_labels = []
        successful_true_labels = []
        successful_adversarials = []

        for genuine, label, true_label, adversarial in zip(self.genuines, self.labels, self.true_labels, self.adversarials):
            if adversarial is not None:
                successful_genuines.append(genuine)
                successful_labels.append(label)
                successful_true_labels.append(true_label)
                successful_adversarials.append(adversarial)

        return successful_genuines, successful_labels, successful_true_labels, successful_adversarials

# TODO: Add/rewrite AdversarialTrainingDataset
    """
    def to_adversarial_training_dataset(self, use_true_labels):
        _, successful_labels, successful_true_labels, successful_adversarials = self.successful_adversarials()

        if use_true_labels:
            used_labels = successful_true_labels
        else:
            used_labels = successful_labels

        return AdversarialTrainingDataset(successful_adversarials, used_labels)
    """

    def index_of_genuine(self, genuine, rtol=1e-5, atol=1e-8):
        genuine = genuine.cpu()

        for i in range(len(self)):
            # Using NumPy's isclose() equation
            if (torch.abs(genuine - self.genuines[i]) <= atol + rtol * torch.abs(self.genuines[i])).all():
                return i

        return -1

    def index_of_genuines(self, genuines, rtol=1e-5, atol=1e-8):
        return [self.index_of_genuine(x, rtol=rtol, atol=atol) for x in genuines]

    def __getitem__(self, idx):
        return (self.genuines[idx], self.labels[idx], self.true_labels[idx], self.adversarials[idx])

    def __len__(self):
        return len(self.genuines)

    def print_stats(self):
        distances = self.successful_distances.numpy()

        success_rate = self.attack_success_rate
        median_distance = np.median(distances)
        average_distance = np.average(distances)

        print('Success Rate: {:.2f}%'.format(success_rate * 100.0))
        print(f'Median Successful Distance: {median_distance}')
        print(f'Average Successful Distance: {average_distance}')

class MIPDataset(data.Dataset):
    def __init__(self, genuines, true_labels, adversarials, lower_bounds, upper_bounds, solve_times, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs):
        assert len(genuines) == len(true_labels)
        assert len(genuines) == len(adversarials)
        assert len(genuines) == len(lower_bounds)
        assert len(genuines) == len(upper_bounds)
        assert len(genuines) == len(solve_times)
        assert all([upper is None or lower is None or upper >=
                    lower for upper, lower in zip(upper_bounds, lower_bounds)])

        self.genuines = [genuine.detach().cpu() for genuine in genuines]
        self.true_labels = [true_label.detach().cpu()
                            for true_label in true_labels]
        self.adversarials = adversarials
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.solve_times = solve_times
        self.p = p
        self.misclassification_policy = misclassification_policy
        self.attack_configuration = attack_configuration
        self.start = start
        self.stop = stop
        self.generation_kwargs = generation_kwargs

    def __getitem__(self, idx):
        return (self.genuines[idx], self.true_labels[idx], self.adversarials[idx], self.lower_bounds[idx], self.upper_bounds[idx], self.solve_times[idx])

    def __len__(self):
        return len(self.genuines)

    @property
    def absolute_differences(self):
        return [None if (lower is None or upper is None) else upper - lower for upper, lower in zip(self.upper_bounds, self.lower_bounds)]

    def print_stats(self):
        absolute_differences = self.absolute_differences
        successful_absolute_differences = np.array(
            [x for x in absolute_differences if x is None])

        print('Median Successful Absolute Differences: ',
              np.median(successful_absolute_differences))
        print('Average Successful Absolute Differences: ',
              np.average(successful_absolute_differences))
        print('Convergence stats:\n')
        for solve_time, absolute_difference in self.convergence_stats:
            print(f'Solve time: {solve_time}, absolute difference: {absolute_difference}')

    @property
    def convergence_stats(self):
        return list(zip(self.solve_times, self.absolute_differences))


class AdversarialDistanceDataset(data.Dataset):
    def __init__(self, images, distances):
        assert len(images) == len(distances)

        self.images = images
        self.distances = distances

    def __getitem__(self, idx):
        return (self.images[idx], self.distances[idx])

    def __len__(self):
        return len(self.images)

class AttackComparisonDataset(data.Dataset):
    def __init__(self, genuines, labels, true_labels, attack_names, attack_results, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs):
        assert len(genuines) == len(labels)
        assert len(genuines) == len(true_labels)
        assert len(genuines) == len(attack_results)

        self.genuines = [genuine.detach().cpu() for genuine in genuines]
        self.labels = [label.detach().cpu() for label in labels]
        self.true_labels = [true_label.detach().cpu()
                            for true_label in true_labels]
        self.attack_names = attack_names
        self.attack_results = attack_results

        # Detach and convert to CPU each adversarial example
        for attack_result in self.attack_results:
            for key, value in attack_result.items():
                if value is not None:
                    attack_result[key] = value.detach().cpu()

        self.p = p
        self.misclassification_policy = misclassification_policy
        self.attack_configuration = attack_configuration
        self.start = start
        self.stop = stop
        self.generation_kwargs = generation_kwargs

    def __getitem__(self, idx):
        return (self.genuines[idx], self.labels[idx], self.true_labels[idx], self.attack_results[idx])

    def __len__(self):
        return len(self.genuines)

    def to_adversarial_dataset(self, attack_name):
        if attack_name not in self.attack_names:
            raise ValueError(f'attack_name must be one of {self.attack_names}.')
        adversarials = [attack_result[attack_name]
                        for attack_result in self.attack_results]
        return AdversarialDataset(self.genuines, self.labels, self.true_labels, adversarials, self.p, self.misclassification_policy, self.attack_configuration, self.start, self.stop, self.generation_kwargs)

    def simulate_pooling(self, selected_attacks):
        # Note: if multiple results have the same distance, it returns
        # the one whose attack appears first in the list of selected attacks
        best_adversarials = []

        for genuine, attack_result in zip(self.genuines, self.attack_results):
            # Take the adversarials generated by attacks that were successful and
            # were selected by the user
            successful_adversarials = [attack_result[attack_name]
                                       for attack_name in selected_attacks if attack_result[attack_name] is not None]

            if len(successful_adversarials) > 0:
                successful_adversarials = torch.stack(successful_adversarials)

                distances = utils.one_many_adversarial_distance(
                    genuine, successful_adversarials, self.p)
                assert distances.shape == (len(successful_adversarials),)

                # torch.argmin returns the first index with minimal value
                best_distance_index = torch.argmin(distances)

                best_adversarials.append(
                    successful_adversarials[best_distance_index])
            else:
                best_adversarials.append(None)

        assert len(self.genuines) == len(best_adversarials)

        return AdversarialDataset(self.genuines, self.labels, self.true_labels, best_adversarials, self.p, self.misclassification_policy, self.attack_configuration, self.start, self.stop, self.generation_kwargs)

    def attack_ranking_stats(self, attack_name):
        attack_positions = dict()
        for i in range(len(self.attack_names)):
            attack_positions[i] = 0
            attack_positions[str(i) + '_ex_aequo'] = 0

        # "failure" represents cases where all attacks failed
        attack_positions['failure'] = 0

        for genuine, attack_result in zip(self.genuines, self.attack_results):
            if attack_result[attack_name] is None:
                attack_positions['failure'] += 1
            else:
                attack_result = [(image, adversarial) for image, adversarial in attack_result.items()
                                 if adversarial is not None]

                # Dictionaries don't preserve order, so we convert to OrderedDict
                attack_result = collections.OrderedDict(attack_result)

                adversarials = torch.stack(list(attack_result.values()))
                distances = utils.one_many_adversarial_distance(
                    genuine, adversarials, self.p)

                assert len(attack_result) == len(distances)

                attack_distance_pairs = list(
                    zip(attack_result.keys(), distances))

                # Sort by distance in ascending order (lower distance = better)
                # If two attacks have the same distance, priority is given to the
                # queried attack
                ordering = lambda pair: (pair[1], 0 if pair[0] == attack_name else 1)
                attack_distance_pairs = sorted(
                    attack_distance_pairs, key=ordering)

                sorted_test_names = [x[0] for x in attack_distance_pairs]

                attack_ranking = sorted_test_names.index(attack_name)
                attack_distance = attack_distance_pairs[attack_ranking][1]

                # Check if the result was ex aequo
                same_distance = [torch.eq(distance, attack_distance) for distance in distances]
                assert sum(same_distance) >= 1

                ex_aequo = sum(same_distance) >= 2

                if ex_aequo:
                    attack_positions[str(attack_ranking) + '_ex_aequo'] += 1
                else:
                    attack_positions[attack_ranking] += 1

        assert sum(count for count in attack_positions.values()
                   ) == len(self.genuines)

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

        for genuine, attack_result in zip(self.genuines, self.attack_results):
            # Dictionaries don't preserve order, so we convert to OrderedDict
            attack_result = collections.OrderedDict(attack_result)

            successful_attacks = [
                name for name in self.attack_names if attack_result[name] is not None]
            unsuccessful_attacks = [
                name for name in self.attack_names if attack_result[name] is None]

            # Successful attacks always beat unsuccessful attacks
            for successful_attack in successful_attacks:
                for unsuccessful_attack in unsuccessful_attacks:
                    victory_matrix[successful_attack][unsuccessful_attack] += 1

            if len(successful_attacks) > 0:
                adversarials = [
                    attack_result[name] for name in successful_attacks]
                adversarials = torch.stack(adversarials)
                distances = utils.one_many_adversarial_distance(
                    genuine, adversarials, self.p)

                assert len(successful_attacks) == len(distances)

                attack_distance_pairs = list(
                    zip(successful_attacks, distances))

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
