import click
import json
import os
from pathlib import Path
import sys
sys.path.append('.')

import utils

@click.command()
@click.argument('reference_dataset', type=click.Path(exists=True))
@click.argument('prefix', type=str)
@click.argument('domain', type=str)
@click.argument('attacks', type=str)
@click.argument('output_file', type=click.Path(exists=False))
def main(reference_dataset, prefix, domain, attacks, output_file):
    reference = utils.load_zip(reference_dataset)
    
    indices = list(reference.attack_results.keys())

    content = ''

    for index in indices:
        for architecture in ['a', 'b', 'c']:
            for test in ['standard', 'adversarial', 'relu']:
                command = f'{prefix} {domain} {architecture} {test} {attacks} {index}'
                content += f'{command}\n'

    with open(output_file, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()