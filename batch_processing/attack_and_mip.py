import click
import os
from pathlib import Path
import sys
sys.path.append('.')
import utils

def prepare_path(path):
    # Must not exist
    if path.exists():
        raise RuntimeError(f'{path} must not exist.')

    # Create parents
    path.parent.mkdir(parents=True, exist_ok=True)

def run_and_log(command, log_file):
    if os.name == 'nt': # Windows
        # Replace double quotes with single quotes
        escaped_command = command.replace('"', "'")
        os.system(f'powershell "{escaped_command} 2>&1 | Tee-Object -Filepath \'{log_file}\'"')
    elif os.name == 'posix': # Unix
        os.system(f'{command} 2>&1 | tee "{log_file}"')
    else:
        raise NotImplementedError

@click.command()
@click.argument('domain', type=click.Choice(['cifar10', 'mnist']))
@click.argument('architecture', type=click.Choice(['a', 'b', 'c']))
@click.argument('test_name', type=click.Choice(['standard', 'adversarial', 'relu']))
@click.argument('parameter_set', type=click.Choice(['original', 'default'])) # Original: the one used for the MIP results in the paper; default: the one used for the attack study in the paper
@click.argument('start', type=click.IntRange(0, None))
@click.argument('stop', type=click.IntRange(1, None))
@click.option('--log-dir', type=click.Path(file_okay=False, dir_okay=True), default='logs')
@click.option('--no-log', is_flag=True)
def main(domain, architecture, test_name, parameter_set, start, stop, log_dir, no_log):
    assert stop > start

    create_logs = not no_log

    log_dir = Path(log_dir) / f'{test_name}/{domain}-{architecture}/{start}-{stop}'

    print(f'Attacking {domain} {architecture} ({test_name}, {start}-{stop})')

    dataset = 'std:test'

    attacks = '"[bim, brendel, carlini, deepfool, fast_gradient, pgd, uniform]"'
    p = 'linf'

    # 0 was used during development, using 1 for actual tests
    seed = 1

    # Attacks are run on CPU, so there's no point in using higher batch sizes
    batch_size = 1
    device = 'cpu'
    cpu_threads = 1
    misclassification_policy = 'use_predicted'
    no_stats_argument = '--no-stats'

    if parameter_set == 'original':
        parameter_set_path = 'original_mip_attack_configuration.cfg'
    else:
        parameter_set_path = 'default_attack_configuration.cfg'

    if test_name == 'relu':
        state_dict_path = f'trained-models/classifiers/{test_name}/relu-pruned/{domain}-{architecture}.pth'
        masked_relu_argument = '--masked-relu'
    else:
        state_dict_path = f'trained-models/classifiers/{test_name}/{domain}-{architecture}.pth'
        masked_relu_argument = ''

    compare_results_path = f'comparison_results/{test_name}/{domain}-{architecture}/{start}-{stop}.zip'

    if os.path.exists(compare_results_path):
        print('Skipping Compare')
    else:
        compare_log_file = log_dir / 'compare.log'
        if create_logs:
            prepare_path(compare_log_file)

        compare_command = f'python cli.py compare {domain} {architecture} {dataset} {attacks} {p} '
        compare_command += f'--state-dict-path {state_dict_path} {masked_relu_argument} '
        compare_command += f'--batch-size {batch_size} --device {device} --cpu-threads {cpu_threads} '
        compare_command += f'--misclassification-policy {misclassification_policy} {no_stats_argument} '
        compare_command += f'--start {start} --stop {stop} --save-to {compare_results_path} '
        compare_command += f'--deterministic --seed {seed} '
        compare_command += f'--attack-config-file {parameter_set_path} '

        print(f'Compare | Running command\n{compare_command}')

        if create_logs:
            run_and_log(compare_command, compare_log_file)
        else:
            os.system(compare_command)

    mip_results_path = f'mip_results/{test_name}/{domain}-{architecture}/{start}-{stop}.zip'

    if os.path.exists(mip_results_path):
        print('Skipping MIP')
        mip_results = utils.load_zip(mip_results_path)
        mip_results.print_stats()
    else:
        gurobi_log_dir = log_dir / 'gurobi_logs'
        mip_log_file = log_dir / 'mip.log'
        memory_log_file = log_dir / 'mip_memory.dat'

        if create_logs:
            for path in [gurobi_log_dir, mip_log_file]:
                prepare_path(path)
        # The memory log file is generated in all cases
        prepare_path(memory_log_file)

        mip_command = f'python cli.py mip {domain} {architecture} {dataset} {p} '
        mip_command += f'--state-dict-path {state_dict_path} {masked_relu_argument} '
        mip_command += f'--batch-size {batch_size} --device {device} --cpu-threads {cpu_threads} '
        mip_command += f'--pre-adversarial-dataset {compare_results_path} '
        mip_command += f'--misclassification-policy {misclassification_policy} '
        mip_command += f'--start {start} --stop {stop} --save-to {mip_results_path} '
        mip_command += f'--deterministic --seed {seed} '

        if create_logs:
            mip_command += f'--gurobi-log-dir {gurobi_log_dir} '
        mip_command = f'mprof run --multiprocess --python --output {memory_log_file} ' + mip_command

        print(f'MIP | Running command\n{mip_command}')

        if create_logs:
            run_and_log(mip_command, mip_log_file)
        else:
            os.system(mip_command)

if __name__ == '__main__':
    main()