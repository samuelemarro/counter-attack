import collections

import numpy as np
import torch
import torch.utils.data as data

import utils

DISTANCE_ATOL = 5e-7
MEDIAN_AVERAGE_ATOL = DISTANCE_ATOL

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

    @property
    def successful_count(self):
        return len(self.successful_indices)

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

    @property
    def successful_indices(self):
        return [i for i in range(len(self)) if self.adversarials[i] is not None]

# TODO: Add/rewrite AdversarialTrainingDataset
    """
    def to_adversarial_training_dataset(self, use_true_labels):
        _, successful_labels, successful_true_labels, successful_adversarials = self.successful_adversarials

        if use_true_labels:
            used_labels = successful_true_labels
        else:
            used_labels = successful_labels

        return AdversarialTrainingDataset(successful_adversarials, used_labels)
    """

    def index_of_genuine(self, genuine, rtol=1e-5, atol=1e-8):
        assert len(genuine.shape) == 3
        genuine = genuine.cpu()

        for i in range(len(self)):
            # Using NumPy's isclose() inequality
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
    def __init__(self, genuines, labels, true_labels, adversarials, lower_bounds, upper_bounds, elapsed_times, extra_infos, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs, global_extra_info):
        assert len(genuines) == len(labels)
        assert len(genuines) == len(true_labels)
        assert len(genuines) == len(adversarials)
        assert len(genuines) == len(lower_bounds)
        assert len(genuines) == len(upper_bounds)
        assert len(genuines) == len(elapsed_times)
        assert len(genuines) == len(extra_infos)
        assert all([upper is None or lower is None or upper >=
                    lower for upper, lower in zip(upper_bounds, lower_bounds)])

        self.genuines = [genuine.detach().cpu() for genuine in genuines]
        self.labels = [label.detach().cpu() for label in labels]
        self.true_labels = [true_label.detach().cpu()
                            for true_label in true_labels]
        self.adversarials = adversarials
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.elapsed_times = elapsed_times
        self.extra_infos = extra_infos
        self.p = p
        self.misclassification_policy = misclassification_policy
        self.attack_configuration = attack_configuration
        self.start = start
        self.stop = stop
        self.generation_kwargs = generation_kwargs
        self.global_extra_info = global_extra_info

    def __getitem__(self, idx):
        return (self.genuines[idx], self.labels[idx], self.true_labels[idx], self.adversarials[idx], self.lower_bounds[idx], self.upper_bounds[idx], self.elapsed_times[idx])

    def __len__(self):
        return len(self.genuines)

    @property
    def absolute_differences(self):
        return [None if (lower is None or upper is None) else upper - lower for upper, lower in zip(self.upper_bounds, self.lower_bounds)]

    def print_stats(self):
        absolute_differences = self.absolute_differences
        successful_absolute_differences = np.array(
            [diff for diff in absolute_differences if diff is not None])

        print('Median Successful Absolute Differences: ',
              np.median(successful_absolute_differences))
        print('Average Successful Absolute Differences: ',
              np.average(successful_absolute_differences))

        print('Convergence stats:')
        for elapsed_time, lower, upper in zip(self.elapsed_times, self.lower_bounds, self.upper_bounds):
            print(f'Elapsed time: {elapsed_time}, ', end='')
            if upper is not None and lower is not None:
                print(f'absolute difference: {upper - lower}')
            else:
                print(f'lower: {lower}, upper: {upper}')

        print()

    @property
    def convergence_stats(self):
        return list(zip(self.elapsed_times, self.absolute_differences))

class MergedDataset:
    def __init__(self):
        self.genuines = {}
        self.labels = {}
        self.true_labels = {}
        self.adversarials = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.elapsed_times = {}
        self.extra_infos = {}
        self.global_extra_infos = {}
        self.generation_kwargs = {}
        self.memory_logs = {}

        self.attack_configuration = None
        self.misclassification_policy = None
        self.p = None

class MergedComparisonDataset:
    def __init__(self):
        self.genuines = {}
        self.labels = {}
        self.true_labels = {}
        self.attack_results = {}
        self.generation_kwargs = {}
        self.logs = {}

        self.attack_configuration = None
        self.attack_names = None
        self.misclassification_policy = None
        self.p = None

    def print_stats(self):
        keys = list(self.genuines.keys())
        comparison_dataset = AttackComparisonDataset(
            [self.genuines[key] for key in keys],
            [self.labels[key] for key in keys],
            [self.true_labels[key] for key in keys],
            self.attack_names,
            [self.attack_results[key] for key in keys],
            self.p,
            self.misclassification_policy,
            self.attack_configuration,
            None,
            None,
            self.generation_kwargs
        )

        comparison_dataset.print_stats()


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
    def __init__(self, genuines, labels, true_labels, attack_names, attack_results, p, misclassification_policy, attack_configuration, start, stop, generation_kwargs, indices_override=None):
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
        self.indices_override = indices_override

    def __getitem__(self, idx):
        return (self.genuines[idx], self.labels[idx], self.true_labels[idx], self.attack_results[idx])

    def __len__(self):
        return len(self.genuines)

    def to_adversarial_dataset(self, attack_names):
        return self.simulate_pooling(attack_names)

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

    def attack_ranking_stats(self, attack_name, atol=DISTANCE_ATOL):
        # Note: Some ex aequo results might be treated differently depending on the
        # attack considered. For example, suppose that after a test the resulting
        # distances are:
        # bim: 1.01
        # pgd: 1.00
        # uniform: 0.99
        # and that the equality threshold is 0.005
        # All three will be considered ex aequo, but
        # - bim will be ex aequo with pgd and worse than uniform      --> 2° ex aequo
        # - pgd will be ex aequo with both bim and pgd                --> 1° ex aequo
        # - uniform will be ex aequo with pgd and better than uniform --> 1° ex aequo

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

                attack_distance = distances[self.attack_names.index(attack_name)]

                better_distance_count = np.count_nonzero([distance < attack_distance - atol for distance in distances])
                same_distance_count = np.count_nonzero([np.abs(distance - attack_distance) <= atol for distance in distances])
                worse_distance_count = np.count_nonzero([distance > attack_distance + atol for distance in distances])

                assert same_distance_count >= 1
                assert better_distance_count + same_distance_count + worse_distance_count == len(self.attack_names)

                ex_aequo = same_distance_count > 1

                if ex_aequo:
                    attack_positions[str(better_distance_count) + '_ex_aequo'] += 1
                else:
                    attack_positions[better_distance_count] += 1

        assert sum(count for count in attack_positions.values()
                   ) == len(self.genuines)

        # Convert absolute numbers to relative
        for key in attack_positions.keys():
            attack_positions[key] /= len(self.genuines)

        return attack_positions

    def pairwise_comparison(self, atol=DISTANCE_ATOL):
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
                        # distance (taking numerical precision into account)
                        if winner_distance < loser_distance - atol:
                            assert winner_attack != loser_attack
                            victory_matrix[winner_attack][loser_attack] += 1

        for winner_attack, losers in victory_matrix.items():
            for loser_attack in losers.keys():
                assert 0 <= victory_matrix[winner_attack][loser_attack] <= len(self.genuines)
                assert 0 <= victory_matrix[loser_attack][winner_attack] <= len(self.genuines)
                assert victory_matrix[winner_attack][loser_attack] + victory_matrix[loser_attack][winner_attack] <= len(self.genuines)

        # Convert absolute numbers to relative
        for loser_dict in victory_matrix.values():
            for key2 in loser_dict.keys():
                loser_dict[key2] /= len(self.genuines)

        return victory_matrix

    def print_stats(self,
                    median_average_atol=MEDIAN_AVERAGE_ATOL,
                    attack_ranking_atol=DISTANCE_ATOL,
                    pairwise_comparison_atol=DISTANCE_ATOL):
        print('===Standard Result===')
        complete_pool = self.simulate_pooling(self.attack_names)
        complete_pool.print_stats()
        print()

        # How much does a single attack contribute to the overall quality?
        print('===Attack Dropping Effects===')

        for attack_name in self.attack_names:
            other_attack_names = [x for x in self.attack_names if x != attack_name]
            other_adversarial_dataset = self.simulate_pooling(
                other_attack_names)

            print(f'Without {attack_name}:')

            other_adversarial_dataset.print_stats()
            print()

        attack_powerset = utils.powerset(self.attack_names, False)

        print('===Pool Stats===')
        for attack_set in attack_powerset:
            print(f'Pool {attack_set}:')

            pool_adversarial_dataset = self.simulate_pooling(attack_set)
            pool_adversarial_dataset.print_stats()
            print()

        print()
        print('===Best Pools===')
        print()

        for n in range(1, len(self.attack_names) + 1):
            print(f'==Pool of size {n}==')
            print()

            n_size_sets = [
                subset for subset in attack_powerset if len(subset) == n]
            n_size_pools = [self.simulate_pooling(
                subset) for subset in n_size_sets]

            attack_success_rates = np.array(
                [x.attack_success_rate for x in n_size_pools])
            median_distances = np.array(
                [np.median(x.successful_distances) for x in n_size_pools])
            average_distances = np.array(
                [np.average(x.successful_distances) for x in n_size_pools])

            best_success_rate = np.max(attack_success_rates)
            best_indices_by_success_rate = [i for i in range(len(n_size_pools)) if attack_success_rates[i] == best_success_rate]

            print(f'Best pools of size {n} by success rate:')
            for index in best_indices_by_success_rate:
                print(f'{n_size_sets[index]}:')
                n_size_pools[index].print_stats()
                print('===')
            print()

            best_median_distance = np.min(median_distances)
            best_indices_by_median_distance = [i for i in range(len(n_size_pools))
                if np.abs(median_distances[i] - best_median_distance) < median_average_atol]

            print(f'Best pools of size {n} by successful median distance (atol={median_average_atol}):')
            for index in best_indices_by_median_distance:
                print(f'{n_size_sets[index]}:')
                n_size_pools[index].print_stats()
                print('===')
            print()

            best_average_distance = np.min(average_distances)
            best_indices_by_average_distance = [i for i in range(len(n_size_pools))
                if np.abs(average_distances[i] - best_average_distance) < median_average_atol]

            print(f'Best pools of size {n} by successful average distance (atol={median_average_atol}):')
            for index in best_indices_by_average_distance:
                print(f'{n_size_sets[index]}:')
                n_size_pools[index].print_stats()
                print('===')
            print()

        print('===Attack Ranking Stats===')

        for attack_name in self.attack_names:
            print(f'Attack {attack_name} (atol={attack_ranking_atol}):')

            attack_ranking_stats = self.attack_ranking_stats(attack_name, atol=attack_ranking_atol)

            for position in range(len(self.attack_names)):
                print('The attack is {}°: {:.2f}%'.format(
                    position + 1, attack_ranking_stats[position] * 100.0))
                print('The attack is {}° ex aequo: {:.2f}%'.format(
                    position + 1, attack_ranking_stats[str(position) + '_ex_aequo'] * 100.0))

            print('The attack fails: {:.2f}%'.format(
                attack_ranking_stats['failure'] * 100.0))
            print()

        print()
        print('===One vs One Comparison===')
        print('atol=', pairwise_comparison_atol)

        victory_matrix = self.pairwise_comparison(atol=pairwise_comparison_atol)

        for winner, loser_dict in victory_matrix.items():
            for loser, rate in loser_dict.items():
                print('{} beats {}: {:.2f}%'.format(winner, loser, rate * 100.0))
