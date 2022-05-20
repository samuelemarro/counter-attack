import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats

@click.command()
@click.argument('domain')
def main(domain):
    valid_keys = None

    for architecture in ['a', 'b', 'c']:
        for test_name in ['standard', 'adversarial', 'relu']:
            mip_path = f'analysis/distances/mip/{domain}-{architecture}-{test_name}.json'

            with open(mip_path, 'r') as f:
                mip_distances = json.load(f)

            if valid_keys is None:
                valid_keys = set(mip_distances.keys())
            else:
                valid_keys &= set(mip_distances.keys())

    valid_keys = [int(x) for x in valid_keys]

    print(len(valid_keys))

    with open(f'{domain}_indices_intersection.json', 'w') as f:
        json.dump(list(valid_keys), f)

if __name__ == '__main__':
    main()