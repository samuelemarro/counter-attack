import click
import os

@click.command()
@click.argument('domain', type=click.Choice(['cifar10', 'mnist']))
@click.argument('architecture', type=click.Choice(['a', 'b', 'c']))
@click.argument('test_name', type=click.Choice(['standard', 'adversarial', 'relu']))
@click.argument('start', type=click.IntRange(0, None))
@click.argument('stop', type=click.IntRange(1, None))
def main(domain, architecture, test_name, start, stop):
    assert stop > start

    dataset = 'std:test'
    # TODO: Anche deepfool?
    attacks = '"[bim, brendel, carlini, deepfool, fast_gradient, pgd, uniform]"'
    p = 'linf'
    seed = 0
    # Attacks are run on CPU, so there's no point in using higher batch sizes
    batch_size = 1
    device = 'cpu'
    cpu_threads = 1
    misclassification_policy = 'use_predicted'
    no_stats_argument = '--no-stats'

    if test_name == 'relu':
        state_dict_path = f'trained-models/{test_name}/relu-pruned/{domain}-{architecture}.pth'
        masked_relu_argument = '--masked-relu'
    else:
        state_dict_path = f'trained-models/classifiers/{test_name}/{domain}-{architecture}.pth'
        masked_relu_argument = ''

    compare_results_path = f'comparison_results/{test_name}/{domain}-{architecture}/{start}-{stop}.zip'

    if os.path.exists(compare_results_path):
        print('Skipping Compare')
    else:
        compare_command = f'python cli.py compare {domain} {architecture} {dataset} {attacks} {p} '
        compare_command += f'--state-dict-path {state_dict_path} {masked_relu_argument} '
        compare_command += f'--batch-size {batch_size} --device {device} --cpu-threads {cpu_threads} '
        compare_command += f'--misclassification-policy {misclassification_policy} {no_stats_argument} '
        compare_command += f'--start {start} --stop {stop} --save-to {compare_results_path} '
        compare_command += f'--deterministic --seed {seed} '

        print(f'Compare | Running command\n{compare_command}')
        os.system(compare_command)

    mip_results_path = f'mip_results/{test_name}/{domain}-{architecture}/{start}-{stop}.zip'

    if os.path.exists(mip_results_path):
        print('Skipping MIP')
    else:
        mip_command = f'python cli.py mip {domain} {architecture} {dataset} {p} '
        mip_command += f'--state-dict-path {state_dict_path} {masked_relu_argument} '
        mip_command += f'--batch-size {batch_size} --device {device} --cpu-threads {cpu_threads} '
        mip_command += f'--pre-adversarial-dataset {compare_results_path} '
        mip_command += f'--misclassification-policy {misclassification_policy} '
        mip_command += f'--start {start} --stop {stop} --save-to {mip_results_path} '
        mip_command += f'--deterministic --seed {seed} '

        print(f'MIP | Running command\n{mip_command}')
        os.system(mip_command)

if __name__ == '__main__':
    main()