import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats

@click.command()
def main():
    for domain in ['mnist', 'cifar10']:
        with open(f'{domain}_indices_intersection.json', 'r') as f:
            indices = json.load(f)
        for architecture in ['a', 'b', 'c']:
            for test_name in ['standard', 'adversarial', 'relu']:
                for context in ['strong', 'balanced', 'mip']:
                    distance_path = f'analysis/distances/{context}/{domain}-{architecture}-{test_name}.json'

                    with open(distance_path, 'r') as f:
                        distances = json.load(f)

                    distances = {k: v for k, v in distances.items() if int(k) in indices}
                    print(len(distances))

                    output_path = f'analysis/distances/intersection/{context}/{domain}-{architecture}-{test_name}.json'

                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

                    with open(output_path, 'w') as f:
                        json.dump(distances, f)

if __name__ == '__main__':
    main()