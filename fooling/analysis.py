import sys

sys.path.append('.')

import json
import click

@click.command()
@click.argument('domain')
@click.argument('architecture')
@click.argument('test_type')
@click.argument('coefficient_name')
@click.argument('element_count', type=int)
def main(domain, architecture, test_type, coefficient_name, element_count):
    if int(float(coefficient_name)) == 1:
        folder_path = f'fooling/results/{domain}/{architecture}/{test_type}'
    else:
        folder_path = f'fooling/results/{coefficient_name}/{domain}/{architecture}/{test_type}'

    with open(f'fooling/sorted_indices_{domain}.json') as f:
        indices = json.load(f)
    
    indices = indices[:element_count]

    if domain == 'mnist':
        eps_names = ['0.025', '0.05', '0.1']
        eps_values = [0.025, 0.05, 0.1]
    else:
        eps_names = ['2/255', '4/255', '8/255']
        eps_values = [2/255, 4/255, 8/255]

    success_count = { eps_name : 0 for eps_name in eps_names }

    for index in indices:
        path = f'{folder_path}/{index}.json'
        with open(path) as f:
            results = json.load(f)

        for eps, eps_name in zip(eps_values, eps_names):
            result = results[str(eps)]

            if int(float(coefficient_name)) == 1:
                lambdas = ['10000.0', '100.0', '1.0', '0.01', '0.0001', 'uniform']
                assert isinstance(result, dict)
                assert list(result.keys()) == lambdas
                # print(result)
                success = any(result[lambda_]['success'] for lambda_ in lambdas)

                if success:
                    print(index)
                    print(result)
            else:
                assert isinstance(result, bool)
                success = result
            
            if success:
                success_count[eps_name] += 1

    success_rates = { eps_name : success_count[eps_name] / len(indices) for eps_name in eps_names }

    print(f'==Coefficient {coefficient_name}==')
    for eps_name, success_rate in success_rates.items():
        print(f'{eps_name}: {(success_rate * 100):.2f}%')

if __name__ == '__main__':
    main()