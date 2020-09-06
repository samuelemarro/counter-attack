import click
import matplotlib.pyplot as plt
import numpy as np

import utils

pool_dataset_path = './mnist-mini-100-compare.zip'
perfect_dataset_path = './mnist-mini-100-mip-blind-zero-tolerance.zip'

pool_dataset = utils.load_zip(pool_dataset_path)
perfect_dataset = utils.load_zip(perfect_dataset_path)

perfect_distance_dataset = perfect_dataset.to_distance_dataset(None)


attack_names = pool_dataset.attack_names


print('===Standard Result===')
complete_pool = pool_dataset.simulate_pooling(attack_names)
complete_pool.print_stats()
print()

# How much does a single attack contribute to the overall quality?
print('===Attack Dropping Effects===')

for attack_name in attack_names:
    other_attack_names = [x for x in attack_names if x != attack_name]
    pool_adversarial_dataset = pool_dataset.simulate_pooling(other_attack_names)

    print('Without {}:'.format(attack_name))

    pool_adversarial_dataset.print_stats()
    print()

attack_powerset = utils.powerset(attack_names)


print()
print('===Best Pools===')
print()

best_average_pools = []

for n in range(1, len(attack_names) + 1):
    print('==Pool of size {}=='.format(n))
    print()

    n_size_sets = [subset for subset in attack_powerset if len(subset) == n]
    n_size_pools = [pool_dataset.simulate_pooling(subset) for subset in n_size_sets]

    attack_success_rates = np.array([x.attack_success_rate for x in n_size_pools])
    median_distances = np.array([np.median(x.successful_distances) for x in n_size_pools])
    average_distances = np.array([np.average(x.successful_distances) for x in n_size_pools])

    best_by_success_rate = np.argmax(attack_success_rates)

    print('Best pool of size {} by success rate: {}'.format(n, n_size_sets[best_by_success_rate]))
    n_size_pools[best_by_success_rate].print_stats()
    print()

    best_by_median_distance = np.argmin(median_distances)

    print('Best pool of size {} by successful median distance: {}'.format(n, n_size_sets[best_by_median_distance]))
    n_size_pools[best_by_median_distance].print_stats()
    print()

    best_by_average_distance = np.argmin(average_distances)
    print('Best pool of size {} by successful average distance: {}'.format(n, n_size_sets[best_by_average_distance]))
    n_size_pools[best_by_average_distance].print_stats()
    best_average_pools.append(n_size_sets[best_by_average_distance])
    print()

print('===Attack Ranking Stats===')

for attack_name in attack_names:
    print('Attack {}:'.format(attack_name))

    attack_ranking_stats = pool_dataset.attack_ranking_stats(attack_name)

    for position, rate in [x for x in attack_ranking_stats.items() if x[0] != 'failure']:
        print('The attack is {}°: {:.2f}%'.format(position + 1, rate * 100.0))

    print('The attack fails: {:.2f}%'.format(attack_ranking_stats['failure'] * 100.0))
    print()

print()


best_average_absolute_differences = []
best_sets = []

for n in range(1, len(attack_names) + 1):
    print('==Pool of size {}=='.format(n))
    print()

    n_size_sets = [subset for subset in attack_powerset if len(subset) == n]
    n_size_pools = [pool_dataset.simulate_pooling(subset) for subset in n_size_sets]

    pool_differences = []
    successes = []
    for pool in n_size_pools:
        success_count = 0
        absolute_differences = []
        for (perfect_genuine, perfect_distance), (approximate_genuine, approximate_distance) in zip(perfect_distance_dataset, pool.to_distance_dataset(None)):
            if perfect_distance is None or approximate_distance is None:
                continue
            success_count += 1

            absolute_differences.append(approximate_distance - perfect_distance)

        absolute_differences = np.array(absolute_differences)
        pool_differences.append(absolute_differences)
        successes.append(success_count)

    assert len(pool_differences) == len(successes) == len(n_size_sets) == len(n_size_pools)

    best_success = max(successes)
    for i in range(len(pool_differences)):
        if successes[i] < best_success:
            successes[i] = None
            pool_differences[i] = None
            n_size_sets[i] = None
            n_size_pools[i] = None

    successes = [x for x in successes if x is not None]
    pool_differences = [x for x in pool_differences if x is not None]
    n_size_sets = [x for x in n_size_sets if x is not None]
    n_size_pools = [x for x in n_size_pools if x is not None]

    average_absolute_differences = np.array([np.average(x) for x in pool_differences])
    best_average_absolute_difference_index = np.argmin(average_absolute_differences)

    print('Best pool by adjusted average difference: {} ({})'.format(n_size_sets[best_average_absolute_difference_index], average_absolute_differences[best_average_absolute_difference_index]))
    best_average_absolute_differences.append(average_absolute_differences[best_average_absolute_difference_index])
    best_sets.append(n_size_sets[best_average_absolute_difference_index]) 

plt.plot(np.array(best_average_absolute_differences))
plt.yscale('log')
plt.show()
plt.yscale('linear')

combinations = [[x] for x in pool_dataset.attack_names] + best_sets

for combination in combinations:
    print('Combination: {}'.format(combination))

    approximate_dataset = pool_dataset.simulate_pooling(combination)
    approximate_distance_dataset = approximate_dataset.to_distance_dataset(None)

    absolute_differences = []
    relative_differences = []

    perfect_failures = 0
    approximate_failures = 0
    above_threshold = 0

    for (perfect_genuine, perfect_distance), (approximate_genuine, approximate_distance) in zip(perfect_distance_dataset, approximate_distance_dataset):
        if np.average(np.abs(perfect_genuine - approximate_genuine)) > 1e5:
            raise click.BadArgumentUsage('Datasets don\'t match (different genuine images).')

        if approximate_distance is None:
            approximate_failures += 1
        if perfect_distance is None:
            perfect_failures += 1

        if approximate_distance is None or perfect_distance is None:
            continue

        if approximate_distance < perfect_distance:
            raise click.BadArgumentUsage('Invalid datasets (approximate is better than perfect).')

        if approximate_distance - perfect_distance >= 1/255:
            above_threshold += 1

        absolute_differences.append(approximate_distance - perfect_distance)
        relative_differences.append((approximate_distance - perfect_distance) / perfect_distance)
        
    if len(absolute_differences) == 0:
        above_threshold_rate = np.nan
    else:
        above_threshold_rate = above_threshold / len(absolute_differences)
    print('Greater than or equal to 1/255: {}/{} ({:.2f}%)'.format(above_threshold, len(absolute_differences), above_threshold_rate * 100.0))
    print('Average absolute difference: {:.2e}'.format(np.average(absolute_differences)))
    print('Median absolute difference: {:.2e}'.format(np.median(absolute_differences)))
    print('Average relative difference: {:.2f}%'.format(np.average(relative_differences) * 100.0))
    print('Median relative difference: {:.2f}%'.format(np.median(relative_differences) * 100.0))
    print('Perfect failures: {}'.format(perfect_failures))
    print('Approximate failures: {}'.format(approximate_failures))
    print('===================\n')

    absolute_differences = np.array(absolute_differences)

    if len(absolute_differences) > 0:
        bins = []
        total = 0
        step = 1 / 510
        while total < absolute_differences.max():
            bins.append(total)
            total += step
        bins = np.array(bins)
        plt.hist(absolute_differences, bins=bins)
        plt.show()

best_average_differences = []
for best_average_pool in best_average_pools:
    approximate_dataset = pool_dataset.simulate_pooling(best_average_pool)
    approximate_distance_dataset = approximate_dataset.to_distance_dataset(None)

    absolute_differences = []

    perfect_failures = 0
    approximate_failures = 0
    above_threshold = 0

    for (perfect_genuine, perfect_distance), (approximate_genuine, approximate_distance) in zip(perfect_distance_dataset, approximate_distance_dataset):
        if np.average(np.abs(perfect_genuine - approximate_genuine)) > 1e5:
            raise click.BadArgumentUsage('Datasets don\'t match (different genuine images).')

        if approximate_distance is None:
            approximate_failures += 1
        if perfect_distance is None:
            perfect_failures += 1

        if approximate_distance is None or perfect_distance is None:
            continue

        if approximate_distance < perfect_distance:
            raise click.BadArgumentUsage('Invalid datasets (approximate is better than perfect).')

        if approximate_distance - perfect_distance >= 1/255:
            above_threshold += 1

        absolute_differences.append(approximate_distance - perfect_distance)


