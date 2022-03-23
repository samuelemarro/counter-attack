import sys

sys.path.append('.')

import click
import numpy as np
import torch

import utils

def linf_distance(genuine, adversarial):
    return torch.max(torch.abs(genuine - adversarial))

def best_linf_distance(genuine, adversarials):
    assert not all(adversarial is None for adversarial in adversarials)
    return min([linf_distance(genuine, adversarial) for adversarial in adversarials if adversarial is not None])

def upper_higher_than_lower(dataset, atol=1e-6):
    violations = {}
    for key in dataset.upper_bounds.keys():
        if dataset.lower_bounds[key] is not None and \
           dataset.upper_bounds[key] is not None and \
              dataset.upper_bounds[key] < dataset.lower_bounds[key] - atol:
            violations[key] = (dataset.lower_bounds[key], dataset.upper_bounds[key])
    return violations

def overall_success_rates(dataset):
    pass

def comparison_success_rate(comparison_dataset):
    results = {}

    for name in comparison_dataset.attack_names:
        results[name] = 0
    
    for attack_results in comparison_dataset.attack_results.values():
        for attack_name, result in attack_results.items():
            if result is not None:
                results[attack_name] += 1
    
    for name in comparison_dataset.attack_names:
        results[name] /= len(comparison_dataset.attack_results)
    
    return results

def gurobi_success_rate(mip_dataset, high_distance=1e70):
    # A run is successful if it finds an upper bound
    success = 0
    total = 0

    for key in mip_dataset.upper_bounds.keys():
        if mip_dataset.upper_bounds[key] is not None and mip_dataset.upper_bounds[key] < high_distance:
            success += 1
        total += 1
    
    return success / total

def gurobi_bounds_rate(mip_dataset, high_distance=1e70):
    success = 0
    total = 0

    for key in mip_dataset.upper_bounds.keys():
        if mip_dataset.upper_bounds[key] is not None and mip_dataset.lower_bounds[key] is not None and mip_dataset.upper_bounds[key] < high_distance:
            success += 1
        total += 1
    
    return success / total

def gurobi_optimality_rate(mip_dataset, atol=1e-5, high_distance=1e70):
    success = 0
    total = 0

    for key in mip_dataset.upper_bounds.keys():
        if mip_dataset.lower_bounds[key] is not None and \
            mip_dataset.upper_bounds[key] is not None and \
            mip_dataset.upper_bounds[key] < high_distance and \
            np.abs(mip_dataset.upper_bounds[key] - mip_dataset.lower_bounds[key]) < atol:
            success += 1
        total += 1
    
    return success / total

def gurobi_upper_is_best_attack(mip_dataset, comparison_dataset, atol=1e-6, high_distance=1e70, optimal_only=False, optimality_atol=1e-5):
    violations = {}
    for key in mip_dataset.upper_bounds.keys():
        if mip_dataset.upper_bounds[key] is not None and mip_dataset.upper_bounds[key] < high_distance and comparison_dataset.attack_results[key] is not None:

            if not optimal_only or mip_dataset.lower_bounds[key] is not None and np.abs(mip_dataset.upper_bounds[key] - mip_dataset.lower_bounds[key]) < optimality_atol:
                best_distance = best_linf_distance(mip_dataset.genuines[key], comparison_dataset.attack_results[key].values())

                if best_distance < mip_dataset.upper_bounds[key] - atol:
                    violations[key] = (mip_dataset.upper_bounds[key], best_distance.item())
    return violations

def gurobi_lower_is_best_attack(mip_dataset, comparison_dataset, atol=1e-6, high_distance=1e70):
    violations = {}
    for key in mip_dataset.lower_bounds.keys():
        if mip_dataset.lower_bounds[key] is not None and mip_dataset.lower_bounds[key] < high_distance and comparison_dataset.attack_results[key] is not None:
            best_distance = best_linf_distance(mip_dataset.genuines[key], comparison_dataset.attack_results[key].values())

            if best_distance < mip_dataset.lower_bounds[key] - atol:
                violations[key] = (mip_dataset.lower_bounds[key], best_distance)
    return violations

@click.command()
@click.argument('domain')
@click.argument('architecture')
@click.argument('test')
def main(domain, architecture, test):
    print('=' * 80)
    print(domain, architecture, test)
    print('=' * 80)

    mip_path = f'final-no-extra/{domain}-{architecture}-{test}.zip'
    comparison_path = f'final-comparison/{domain}-{architecture}-{test}.zip'
    print('Loading MIP dataset...')
    mip_dataset = utils.load_zip(mip_path)
    print('Done.')
    print('Loading comparison dataset...')
    comparison_dataset = utils.load_zip(comparison_path)
    print('Done.')

    print('Comparison success rate:')
    comparison_success_rates = comparison_success_rate(comparison_dataset)
    for key, value in comparison_success_rates.items():
        print(f'{key} : {value * 100.}%')

    print(f'Gurobi success rate: {gurobi_success_rate(mip_dataset) * 100.}%')
    print(f'Gurobi bounds rate: {gurobi_bounds_rate(mip_dataset) * 100.}%')
    print(f'Gurobi optimality rate: {gurobi_optimality_rate(mip_dataset) * 100.}%')

    print('=' * 80)
    print('Upper higher than lower')
    violations = upper_higher_than_lower(mip_dataset)

    print('#violations:', len(violations))
    print('Violations:')
    for key, (lower, upper) in violations.items():
        print(f'{key}: {lower} < {upper}')
    if len(violations) > 0:
        print('Worst violation:', max([torch.abs(lower - upper) for lower, upper in violations.values()]))

    print('=' * 80)
    print('Gurobi upper is best attack')
    violations = gurobi_upper_is_best_attack(mip_dataset, comparison_dataset)

    print('#violations:', len(violations))
    """print('Violations:')
    for key, (gurobi_upper, best_distance) in violations.items():
        print(f'{key}:')
        print('Gurobi upper:   ', gurobi_upper)
        print('Best distance:  ', best_distance)"""
    if len(violations) > 0:
        # print()
        print('Worst violation:', max([np.abs(lower - upper) for lower, upper in violations.values()]))
        print('Worst violation (%):', max([(lower - upper) / lower * 100 for lower, upper in violations.values()]))

    print('=' * 80)
    print('Gurobi upper is best attack (optimal only)')
    violations = gurobi_upper_is_best_attack(mip_dataset, comparison_dataset, optimal_only=True)

    print('#violations:', len(violations))
    if len(violations) > 0:
        # print()
        print('Worst violation:', max([np.abs(lower - upper) for lower, upper in violations.values()]))
        print('Worst violation (%):', max([(lower - upper) / lower * 100 for lower, upper in violations.values()]))

    print('=' * 80)
    print('Gurobi lower is best attack')
    violations = gurobi_lower_is_best_attack(mip_dataset, comparison_dataset)

    print('#violations:', len(violations))
    print('Violations:')
    for key, (lower, upper) in violations.items():
        print(f'{key}: {lower} < {upper}')
    if len(violations) > 0:
        print('Worst violation:', max([torch.abs(lower - upper) for lower, upper in violations.values()]))

    print('\n' * 3)

if __name__ == '__main__':
    main()
