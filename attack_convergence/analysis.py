import sys
sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np

def analyze_stats(result, track_every, cutoff, distance_reference, atol=1e-5, rtol=1e-10):
    steps = [1] + list(range(track_every, cutoff, track_every)) + [cutoff]
    distances = { step: [] for step in steps }
    success_count = { step: 0 for step in steps }

    def valid_bounds(lower, upper):
        if lower is None or upper is None:
            return False
        return upper - lower < atol or (upper - lower) / upper < rtol

    valid_keys = [ key for key in result.keys() if valid_bounds(distance_reference[key]['lower'], distance_reference[key]['upper'])]
    print('Dropped', len(result) - len(valid_keys), 'out of', len(result))

    for index in valid_keys:
        for step in steps:
            last_stat = [stat for stat in result[index]['stats'] if stat[0] <= step][-1]
            _, distance = last_stat
            if distance is not None:
                success_count[step] += 1
                distances[step].append(distance - distance_reference[index]['upper'])
    
    success_rates = { step: success_count[step] / len(valid_keys) for step in steps }
    mean_distances = { step: (None if len(distances[step]) == 0 else np.mean(distances[step])) for step in steps }

    return { step: {'success_rate' : success_rates[step], 'distance' : mean_distances[step]} for step in steps }

def to_csv(json_results, attack_names):
    steps = sorted(json_results[attack_names[0]].keys())

    # Create the header
    csv = 'step'
    for attack_name in attack_names:
        csv += f';{attack_name};{attack_name}_partial'
    csv += '\n'

    # Add the data
    for step in steps:
        line_elements = [str(step)]
        for attack_name in attack_names:
            success_rate = json_results[attack_name][step]['success_rate']
            distance = json_results[attack_name][step]['distance']

            if success_rate < 1:
                assert distance is not None or success_rate == 0
                line_elements += ['', '' if success_rate == 0 else str(distance)]
            else:
                assert distance is not None
                line_elements += [str(distance), '']
        
        csv += ';'.join(line_elements) + '\n'
    
    return csv

@click.command()
@click.argument('domain', type=click.Choice(['mnist', 'cifar10']))
@click.argument('architecture', type=click.Choice(['a', 'b', 'c']))
@click.argument('test_type', type=click.Choice(['standard', 'adversarial', 'relu']))
@click.argument('track_every', type=click.IntRange(1, None))
@click.argument('cutoff', type=int)
def main(domain, architecture, test_type, track_every, cutoff):
    attack_names = ['bim', 'brendel', 'carlini', 'deepfool', 'fast_gradient', 'pgd', 'uniform']

    reference_path = f'analysis/distances/mip/{domain}-{architecture}-{test_type}.json'

    with open(reference_path) as f:
        distance_reference = json.load(f)

    for override_set_number in [100, 1000, 10000]:
        analysis_result_path = f'attack_convergence/analysis_results/json/{domain}/{architecture}/{test_type}/{override_set_number}.json'
        csv_result_path = f'attack_convergence/analysis_results/csv/{domain}/{architecture}/{test_type}/{override_set_number}.csv'
        analysis_result_path = Path(analysis_result_path)
        csv_result_path = Path(csv_result_path)
        if not analysis_result_path.exists() or not csv_result_path.exists():
            analysis_result_path.parent.mkdir(parents=True, exist_ok=True)
            csv_result_path.parent.mkdir(parents=True, exist_ok=True)
            global_results = {}
            for attack_name in attack_names:
                with open(f'attack_convergence/attack_results/{domain}/{architecture}/{test_type}/{attack_name}/{override_set_number}.json') as f:
                    result = json.load(f)
                global_results[attack_name] = analyze_stats(result, track_every, cutoff, distance_reference)

            with open(analysis_result_path, 'w') as f:
                json.dump(global_results, f)

            with open(csv_result_path, 'w') as f:
                f.write(to_csv(global_results, attack_names))
    
if __name__ == '__main__':
    main()