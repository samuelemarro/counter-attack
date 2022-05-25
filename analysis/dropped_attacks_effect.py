import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats

import utils

EIGHT_BIT_DISTANCE = 1/255

ATTACK_NAME_TO_DISPLAY_NAME = {
    'none': 'None',
    'bim': 'BIM',
    'brendel': 'B\\&B',
    'carlini': 'C\\&W',
    'deepfool': 'Deepfool',
    'fast_gradient': 'FGSM',
    'pgd': 'PGD',
    'uniform': 'Uniform',
}


def get_r2(domain, parameter_set, atol, rtol, test_override, attacks):
    all_success_rates = []
    all_differences_between_averages = []
    all_r2s = []
    all_difference_below_eight_bit_rates = []

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
                difference_below_eight_bit_rate = 0
            else:
                success_rate = len(target_distances) / num_valid_inputs
                average_target = np.average(target_distances)
                average_approximation = np.average(approximation_distances)
                difference_between_averages = (average_approximation - average_target) / average_target
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_distances, approximation_distances)
                num_difference_below_eight_bit = len([target_distance for target_distance, approximation_distance in zip(target_distances, approximation_distances) \
                    if np.abs(target_distance - approximation_distance) < EIGHT_BIT_DISTANCE])
            
                difference_below_eight_bit_rate = num_difference_below_eight_bit / len(target_distances)

            all_success_rates.append(success_rate)
            all_differences_between_averages.append(difference_between_averages)
            all_r2s.append(r_value ** 2)
            all_difference_below_eight_bit_rates.append(difference_below_eight_bit_rate)


    return (np.mean(all_success_rates), np.std(all_success_rates)), (np.mean(all_differences_between_averages), np.std(all_differences_between_averages)), (np.mean(all_r2s), np.std(all_r2s)), (np.mean(all_difference_below_eight_bit_rates), np.std(all_difference_below_eight_bit_rates))

def pool_selector(pool):
    (success_mean, _), (_, _), (r2_mean, _), (_, _) = pool
    return (success_mean, r2_mean)

ALL_ATTACKS = ['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform']

@click.command()
@click.argument('domain')
@click.argument('parameter_set')
@click.argument('atol', type=float)
@click.argument('rtol', type=float)
@click.option('--test-override', type=str, default=None)
def main(domain, parameter_set, atol, rtol, test_override):
    csv = 'Dropped Attack,Success Rate,Difference,\% Below 1/255,$R^2$\n'

    for attack in ['none'] + ALL_ATTACKS:
        remaining_attacks = [a for a in ALL_ATTACKS if a != attack]
        pool_result = get_r2(domain, parameter_set, atol, rtol, test_override, remaining_attacks)

        (success_rate_mean, success_rate_std), (difference_mean, difference_std), (r2_mean, r2_std), (below_eight_mean, below_eight_std) = pool_result

        csv += f'{ATTACK_NAME_TO_DISPLAY_NAME[attack]},{success_rate_mean * 100:.2f}\\textpm{success_rate_std * 100:.2f}\%,{difference_mean * 100:.2f}\\textpm{difference_std * 100:.2f}\%,{below_eight_mean * 100:.2f}\\textpm{below_eight_std * 100:.2f}\%,{r2_mean:.3f}\\textpm{r2_std:.3f}\n'

    csv_path = f'analysis/dropped-attacks/{domain}-{parameter_set}.csv'

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    Path(csv_path).write_text(csv)

if __name__ == '__main__':
    main()