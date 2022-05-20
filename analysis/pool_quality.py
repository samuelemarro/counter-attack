import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats

import parsing
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
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_distances, approximation_distances)

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
@click.argument('attacks', callback=parsing.ParameterList())
@click.option('--test-override', type=str, default=None)
def main(domain, parameter_set, atol, rtol, attacks, test_override):
    pool_result = get_r2(domain, parameter_set, atol, rtol, test_override, attacks)
    print(pool_result)

if __name__ == '__main__':
    main()