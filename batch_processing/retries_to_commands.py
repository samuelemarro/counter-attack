import click
import json
import os
from pathlib import Path
import sys
sys.path.append('.')

import utils

@click.command()
@click.argument('failure_path', type=click.Path(exists=True))
@click.argument('prefix', type=str)
@click.argument('attack_config_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(exists=False))
def main(failure_path, prefix, attack_config_file, output_file):
    with open(failure_path, 'r') as f:
        failures = json.load(f)
    
    content = ''

    for domain in ['mnist', 'cifar10']:
        for architecture in ['a', 'b', 'c']:
            for test in ['standard', 'adversarial', 'relu']:
                if domain in failures and architecture in failures[domain] and test in failures[domain][architecture]:
                    for index, attacks in failures[domain][architecture][test].items():
                        attack_string = '"[' +  ', '.join(attacks) + ']"'
                        content += f'{prefix} {domain} {architecture} {test} {attack_string} {index} {attack_config_file}\n'

    with open(output_file, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()