import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats

EIGHT_BIT_DISTANCE = 1/255

@click.command()
@click.argument('domain')
@click.argument('parameter_set')
@click.argument('atol', type=float)
@click.argument('rtol', type=float)
@click.option('--test-override', type=str, default=None)
def main(domain, parameter_set, atol, rtol, test_override):
    all_success_rates = []
    all_difference_below_eight_bit_rates = []
    all_differences_between_averages = []
    all_r2s = []

    for test in ['standard', 'adversarial', 'relu'] if test_override is None else [test_override]:
        for architecture in ['a', 'b', 'c']:
            print('=' * 80)
            print(domain, architecture, test)
            print('=' * 80)
            print()

            approximation_path = f'analysis/distances/{parameter_set}/{domain}-{architecture}-{test}.json'

            with open(approximation_path, 'r') as f:
                approximations = json.load(f)
            
            mip_path = f'analysis/distances/mip/{domain}-{architecture}-{test}.json'

            with open(mip_path, 'r') as f:
                mip_distances = json.load(f)

            assert set(mip_distances.keys()) == set(approximations.keys())

            target_distances = []
            approximation_distances = []
            num_valid_inputs = 0

            for index, distances in approximations.items():
                if index not in mip_distances:
                    raise RuntimeError
                upper = mip_distances[index]['upper']
                lower = mip_distances[index]['lower']

                if upper is None or lower is None and not (np.abs(upper - lower) <= atol or np.abs((upper - lower) / upper) <= rtol):
                    continue

                num_valid_inputs += 1

                valid_distances = [distance for distance in distances.values() if distance is not None]

                if len(valid_distances) == 0:
                    continue

                target_distances.append(upper)
                approximation_distances.append(min(valid_distances))

            assert len(target_distances) == len(approximation_distances)

            success_rate = len(target_distances) / num_valid_inputs
            all_success_rates.append(success_rate)
            print(f'{len(target_distances)}/{num_valid_inputs} (success rate: {success_rate * 100.0}%)')

            num_difference_below_eight_bit = len([target_distance for target_distance, approximation_distance in zip(target_distances, approximation_distances) \
                if np.abs(target_distance - approximation_distance) < EIGHT_BIT_DISTANCE])
            
            difference_below_eight_bit_rate = num_difference_below_eight_bit / len(target_distances)
            all_difference_below_eight_bit_rates.append(difference_below_eight_bit_rate)
            print(f'Difference below 8 bit: {num_difference_below_eight_bit}/{len(target_distances)} ({difference_below_eight_bit_rate * 100.0}%)')

            average_target = np.average(target_distances)
            average_approximation = np.average(approximation_distances)
            difference_between_averages = (average_approximation - average_target) / average_target
            all_differences_between_averages.append(difference_between_averages)

            print('Average target distance:', average_target)
            print('Average approximation distance:', average_approximation)
            print(f'Difference: {difference_between_averages * 100.0}%')

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_distances, approximation_distances)
            all_r2s.append(r_value ** 2)

            print('Slope: ', slope)
            print('Intercept: ', intercept)
            print('R2: ', r_value ** 2)

    print('=' * 80)
    print('Success rate:')
    print('Min:', min(all_success_rates))
    print('Max:', max(all_success_rates))
    print('Mean:', np.mean(all_success_rates))
    print('Std:', np.std(all_success_rates))

    print('=' * 80)
    print('Difference below 8 bit:')
    print('Min:', min(all_difference_below_eight_bit_rates))
    print('Max:', max(all_difference_below_eight_bit_rates))
    print('Mean:', np.mean(all_difference_below_eight_bit_rates))
    print('Std:', np.std(all_difference_below_eight_bit_rates))

    print('=' * 80)
    print('Difference between averages:')
    print('Min:', min(all_differences_between_averages))
    print('Max:', max(all_differences_between_averages))
    print('Mean:', np.mean(all_differences_between_averages))
    print('Std:', np.std(all_differences_between_averages))

    print('=' * 80)
    print('R2:')
    print('Min:', min(all_r2s))
    print('Max:', max(all_r2s))
    print('Mean:', np.mean(all_r2s))
    print('Std:', np.std(all_r2s))


if __name__ == '__main__':
    main()