import sys

sys.path.append('.')

import json

import click
import numpy as np
import scipy.stats

import utils

EIGHT_BIT_DISTANCE = 1/255

def get_r2(domain, parameter_set, atol, rtol, test_override, attacks):
    all_success_rates = []
    all_differences_between_averages = []
    all_r2s = []

    for test in ['standard', 'adversarial', 'relu'] if test_override is None else [test_override]:
        for architecture in ['a', 'b', 'c']:
            approximation_path = f'analysis/distances/{parameter_set}/{domain}-{architecture}-{test}.json'

            with open(approximation_path, 'r') as f:
                approximations = json.load(f)
            
            mip_path = f'analysis/distances/mip/{domain}-{architecture}-{test}.json'

            with open(mip_path, 'r') as f:
                mip_distances = json.load(f)

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

                valid_distances = [distances[attack_name] for attack_name in distances.keys() if distances[attack_name] is not None and attack_name in attacks]

                if len(valid_distances) == 0:
                    continue

                target_distances.append(upper)
                approximation_distances.append(min(valid_distances))
            
            if len(target_distances) == 0:
                success_rate = 0
                r_value = 0
                difference_between_averages = np.inf
            else:
                success_rate = len(target_distances) / num_valid_inputs
                average_target = np.average(target_distances)
                average_approximation = np.average(approximation_distances)
                difference_between_averages = (average_approximation - average_target) / average_target
                _, _, r_value, _, _ = scipy.stats.linregress(target_distances, approximation_distances)

            all_success_rates.append(success_rate)
            all_differences_between_averages.append(difference_between_averages)
            all_r2s.append(r_value ** 2)


    return (np.mean(all_success_rates), np.std(all_success_rates)), (np.mean(all_differences_between_averages), np.std(all_differences_between_averages)), (np.mean(all_r2s), np.std(all_r2s))

def pool_selector(pool):
    (success_mean, _), (_, _), (r2_mean, _) = pool
    return (success_mean, r2_mean)

@click.command()
@click.argument('domain')
@click.argument('parameter_set')
@click.argument('atol', type=float)
@click.argument('rtol', type=float)
@click.argument('min_success_rate', type=float)
@click.argument('min_r2', type=float)
@click.option('--test-override', type=str, default=None)
def main(domain, parameter_set, atol, rtol, min_success_rate, min_r2, test_override):
    valid_pools = []
    for attack_set in utils.powerset(['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform'], False):
        pool_result = get_r2(domain, parameter_set, atol, rtol, test_override, attack_set)
        (success_mean, _), (_, _), (r2_mean, _) = pool_result

        if success_mean >= min_success_rate and r2_mean >= min_r2:
            valid_pools.append((attack_set, pool_result))

    min_size = min([len(attack_set) for attack_set, _ in valid_pools])
    valid_pools = [(attack_set, pool) for attack_set, pool in valid_pools if len(attack_set) == min_size]

    valid_pools.sort(key=lambda x: pool_selector(x[1]), reverse=True)
    print(valid_pools)

if __name__ == '__main__':
    main()