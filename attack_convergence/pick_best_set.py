import sys
sys.path.append('.')

import json
from pathlib import Path

import click

@click.command()
@click.argument('domain', type=click.Choice(['mnist', 'cifar10']))
def main(domain):
    analysis_set = [100, 1000, 10000]

    attacks = ['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform']
    best_overrides = {attack: {} for attack in attacks}
    
    for attack_name in attacks:
        analysis_results = { k: [] for k in analysis_set}
        for file in Path(f'./attack_convergence/tuning_results/{domain}/{attack_name}').glob('**/*'):
            if file.is_file():
                with open(str(file)) as f:
                    result = json.load(f)
                
                for step in analysis_set:
                    last_stat = [stat for stat in result['average'] if stat[0] <= step][-1]
                    _, success_rate, best_distance = last_stat
                    analysis_results[step].append((success_rate, best_distance, result))

        for step in analysis_set:
            # Sort by success rate (higher = better) and then distance (lower = better)
            best_set = sorted(analysis_results[step], key=lambda x: (-x[0], x[1]))[0]
            # print('Best', attack_name, best_set[2]['average'][-1][1])
            best_override = best_set[2]['parameters']
            best_overrides[attack_name][step] = best_override
    
    with open(f'attack_convergence/best_overrides_{domain}.json', 'w') as f:
        json.dump(best_overrides, f)

if __name__ == '__main__':
    main()