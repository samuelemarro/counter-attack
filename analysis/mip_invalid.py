import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats


@click.command()
@click.argument('domain')
@click.argument('parameter_set')
@click.argument('atol', type=float)
@click.argument('rtol', type=float)
@click.option('--test-override', type=str, default=None)
def main(domain, parameter_set, atol, rtol, test_override):
    all_valid_rates = []

    for test in ['standard', 'adversarial', 'relu'] if test_override is None else [test_override]:
        for architecture in ['a', 'b', 'c']:
            approximation_path = f'analysis/distances/{parameter_set}/{domain}-{architecture}-{test}.json'

            with open(approximation_path, 'r') as f:
                approximations = json.load(f)
            
            mip_path = f'analysis/distances/mip/{domain}-{architecture}-{test}.json'

            with open(mip_path, 'r') as f:
                mip_distances = json.load(f)

            assert set(mip_distances.keys()) == set(approximations.keys())

            num_valid_inputs = 0

            for index in approximations.keys():
                if index not in mip_distances:
                    raise RuntimeError
                upper = mip_distances[index]['upper']
                lower = mip_distances[index]['lower']

                if upper is None or lower is None and not (np.abs(upper - lower) <= atol or np.abs((upper - lower) / upper) <= rtol):
                    continue

                num_valid_inputs += 1

            valid_rate = num_valid_inputs / len(approximations)

            all_valid_rates.append(valid_rate)

    print('Valid rate:')
    print(np.mean(all_valid_rates))
    print('Invalid rate:')
    print(1 - np.mean(all_valid_rates))


if __name__ == '__main__':
    main()