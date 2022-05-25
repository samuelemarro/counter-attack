import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np

@click.command()
@click.argument('domain')
@click.argument('architecture')
@click.argument('test')
@click.argument('parameter_set')
@click.argument('atol', type=float)
@click.argument('rtol', type=float)
def main(domain, architecture, test, parameter_set, atol, rtol):
    csv = 'True Distance;Estimated Distance\n'
    approximation_path = f'analysis/distances/{parameter_set}/{domain}-{architecture}-{test}.json'

    with open(approximation_path, 'r') as f:
        approximations = json.load(f)

    mip_path = f'analysis/distances/mip/{domain}-{architecture}-{test}.json'

    with open(mip_path, 'r') as f:
        mip_distances = json.load(f)

    assert set(approximations.keys()) == set(mip_distances.keys())

    for index, distances in approximations.items():
        if index not in mip_distances:
            raise RuntimeError
        upper = mip_distances[index]['upper']
        lower = mip_distances[index]['lower']

        if upper is None or lower is None or not (np.abs(upper - lower) <= atol or np.abs((upper - lower) / upper) <= rtol):
            continue

        valid_distances = [distance for distance in distances.values() if distance is not None]

        if len(valid_distances) == 0:
            raise RuntimeError

        best_distance = min(valid_distances)
        csv += f'{upper};{best_distance}\n'

    csv_path = f'analysis/distances/{parameter_set}/csv/{domain}-{architecture}-{test}-{parameter_set}.csv'

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w') as f:
        f.write(csv)

if __name__ == '__main__':
    main()