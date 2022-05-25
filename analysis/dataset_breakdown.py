import sys

sys.path.append('.')

import json
from pathlib import Path

import click
import torch

import parsing

@click.command()
@click.argument('domain')
def main(domain):
    with open(f'{domain}_indices_intersection.json', 'r') as f:
        indices = json.load(f)

    dataset = parsing.parse_dataset(domain, 'std:test')

    counters = { i: 0 for i in range(10) }

    for index in indices:
        _, label = dataset[index]
    
        counters[label] += 1

    csv_path = f'analysis/dataset-breakdown/{domain}.csv'
    csv = 'Label,Count,\%\n'

    for k, v in counters.items():
        csv += f'{k},{v},{v/len(indices)*100:.2f}\%\n'
    
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    Path(csv_path).write_text(csv)



if __name__ == '__main__':
    main()