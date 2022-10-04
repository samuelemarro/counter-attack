import sys
sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats

def analyze_stats(result, track_every, cutoff, distance_reference, atol=1e-5, rtol=1e-10):
    steps = [1] + list(range(track_every, cutoff, track_every)) + [cutoff]
    heuristic_distances = { step: [] for step in steps }
    exact_distances = { step: [] for step in steps }
    success_count = { step: 0 for step in steps }

    def valid_bounds(lower, upper):
        if lower is None or upper is None:
            return False
        return (np.abs(upper - lower) <= atol or np.abs((upper - lower) / upper) <= rtol)

    valid_keys = [ key for key in result.keys() if valid_bounds(distance_reference[key]['lower'], distance_reference[key]['upper'])]
    print('Dropped', len(result) - len(valid_keys), 'out of', len(result))

    for index in valid_keys:
        for step in steps:
            last_stat = [stat for stat in result[index]['stats'] if stat[0] <= step][-1]
            _, distance = last_stat
            if distance is not None:
                success_count[step] += 1
                heuristic_distances[step].append(distance)
                exact_distances[step].append(distance_reference[index]['upper'])
    
    success_rates = { step: success_count[step] / len(valid_keys) for step in steps }

    r_squareds = {}

    for step in steps:
        if success_rates[step] == 0:
            r_squared = None
        else:
            _, _, r_value, _, _ = scipy.stats.linregress(exact_distances[step], heuristic_distances[step])
            r_squared = r_value ** 2

            if np.isnan(r_squared):
                r_squared = None

        r_squareds[step] = r_squared

    return { step: {'success_rate' : success_rates[step], 'r2' : r_squareds[step]} for step in steps }

def to_csv(json_results, attack_names, balanced_r2, strong_r2):
    steps = sorted(json_results[attack_names[0]].keys())

    # Create the header
    csv = 'step'
    for attack_name in attack_names:
        csv += f';{attack_name};{attack_name}_partial'
    csv += ';balanced;strong'
    csv += '\n'

    # Add the data
    for i, step in enumerate(steps):
        line_elements = [str(step)]
        for attack_name in attack_names:
            success_rate = json_results[attack_name][step]['success_rate']
            r2 = json_results[attack_name][step]['r2']

            if success_rate < 1:
                assert r2 is not None or success_rate == 0
                line_elements += ['', '' if success_rate == 0 else str(r2)]
            else:
                assert r2 is not None
                partial_distance = ''
                if i > 0:
                    prev_step = steps[i - 1]
                    prev_success_rate = json_results[attack_name][prev_step]['success_rate']
                    if prev_success_rate < 1:
                        # We've just had a switch from below 1 to 1
                        partial_distance = str(r2)
                line_elements += [str(r2), partial_distance]
        line_elements += [str(balanced_r2), str(strong_r2)]
        csv += ';'.join(line_elements) + '\n'
    
    return csv

def analyze_classic_results(result, distance_reference, atol=1e-5, rtol=1e-10):
    def valid_bounds(lower, upper):
        if lower is None or upper is None:
            return False
        return (np.abs(upper - lower) <= atol or np.abs((upper - lower) / upper) <= rtol)

    valid_keys = [ key for key in result.keys() if valid_bounds(distance_reference[key]['lower'], distance_reference[key]['upper'])]

    heuristic_distances = []
    exact_distances = []
    for index in valid_keys:
        best_distance = min([distance for distance in result[str(index)].values() if distance is not None])
        heuristic_distances.append(best_distance)
        exact_distances.append(distance_reference[index]['upper'])

    _, _, r_value, _, _ = scipy.stats.linregress(exact_distances, heuristic_distances)

    return r_value ** 2

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
        analysis_result_path = f'attack_convergence/analysis_results/r2/json/{domain}/{architecture}/{test_type}/{override_set_number}.json'
        csv_result_path = f'attack_convergence/analysis_results/r2/csv/{domain}/{architecture}/{test_type}/{override_set_number}.csv'
        analysis_result_path = Path(analysis_result_path)
        csv_result_path = Path(csv_result_path)
        if not analysis_result_path.exists() or not csv_result_path.exists():
            analysis_result_path.parent.mkdir(parents=True, exist_ok=True)
            csv_result_path.parent.mkdir(parents=True, exist_ok=True)

            balanced_stats_path = f'analysis/distances/balanced/{domain}-{architecture}-{test_type}.json'
            strong_stats_path = f'analysis/distances/strong/{domain}-{architecture}-{test_type}.json'

            with open(balanced_stats_path) as f:
                balanced_stats = json.load(f)
            with open(strong_stats_path) as f:
                strong_stats = json.load(f)
            
            balanced_r2 = analyze_classic_results(balanced_stats, distance_reference)
            strong_r2 = analyze_classic_results(strong_stats, distance_reference)
            # print(balanced_mean, strong_mean)

            global_results = {}
            for attack_name in attack_names:
                with open(f'attack_convergence/attack_stats/{domain}/{architecture}/{test_type}/{attack_name}/{override_set_number}.json') as f:
                    result = json.load(f)
                global_results[attack_name] = analyze_stats(result, track_every, cutoff, distance_reference)

            with open(analysis_result_path, 'w') as f:
                json.dump(global_results, f)

            with open(csv_result_path, 'w') as f:
                f.write(to_csv(global_results, attack_names, balanced_r2, strong_r2))
    
if __name__ == '__main__':
    main()