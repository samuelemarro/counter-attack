import json
from pathlib import Path
import sys
sys.path.append('.')

import click

import utils

@click.command()
def main():
    failed = {}

    for domain in ['mnist', 'cifar10']:
        if domain not in failed:
            failed[domain] = {}
        for architecture in ['a', 'b', 'c']:
            if architecture not in failed[domain]:
                failed[domain][architecture] = {}
            for test in ['standard', 'adversarial', 'relu']:
                comparison_path = Path('final-comparison') / f'{domain}-{architecture}-{test}.zip'
                comparison_dataset = utils.load_zip(comparison_path)

                failed[domain][architecture][test] = {}

                for index in comparison_dataset.attack_results.keys():
                    for attack_name, attack_result in comparison_dataset.attack_results[index].items():
                        if attack_result is None:
                            if index not in failed[domain][architecture][test]:
                                failed[domain][architecture][test][index] = []
                            failed[domain][architecture][test][index].append(attack_name)
    
    with open('failed_comparisons.json', 'w') as f:
        json.dump(failed, f)
                

if __name__ == '__main__':
    main()