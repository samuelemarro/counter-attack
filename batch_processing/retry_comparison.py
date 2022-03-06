import click
import os
from pathlib import Path
import sys
sys.path.append('.')

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
@click.argument('attacks', type=str)
@click.argument('start', type=click.IntRange(0, None))
@click.argument('attack-config-file', type=click.Path(exists=True))
@click.option('--log-dir', type=click.Path(file_okay=False, dir_okay=True), default='retry_logs')
def main(domain, architecture, test_name, attacks, start, attack_config_file, log_dir):
    stop = start + 1

    log_dir = Path(log_dir) / f'{test_name}/{domain}-{architecture}/{start}-{stop}'
    log_dir.mkdir(parents=True, exist_ok=True)
    Path('comparison_retries').mkdir(parents=True, exist_ok=True)

    print(f'Attacking {domain} {architecture} ({test_name}, {start}-{stop})')

    dataset = 'std:test'

    p = 'linf'

    # 0 was used during development, 1 for tests, using 2 for retries
    seed = 2

    # Attacks are run on CPU, so there's no point in using higher batch sizes
    batch_size = 1
    device = 'cpu'
    cpu_threads = 1
    misclassification_policy = 'use_predicted'
    no_stats_argument = '--no-stats'

    if test_name == 'relu':
        state_dict_path = f'trained-models/classifiers/{test_name}/relu-pruned/{domain}-{architecture}.pth'
        masked_relu_argument = '--masked-relu'
    else:
        state_dict_path = f'trained-models/classifiers/{test_name}/{domain}-{architecture}.pth'
        masked_relu_argument = ''

    compare_results_path = f'comparison_retries/{test_name}/{domain}-{architecture}/{start}-{stop}.zip'

    if os.path.exists(compare_results_path):
        print('Skipping Compare')
    else:
        compare_log_file = log_dir / 'compare.log'
        prepare_path(compare_log_file)

        compare_command = f'python cli.py compare {domain} {architecture} {dataset} {attacks} {p} '
        compare_command += f'--state-dict-path {state_dict_path} {masked_relu_argument} '
        compare_command += f'--batch-size {batch_size} --device {device} --cpu-threads {cpu_threads} '
        compare_command += f'--misclassification-policy {misclassification_policy} {no_stats_argument} '
        compare_command += f'--start {start} --stop {stop} --save-to {compare_results_path} '
        compare_command += f'--deterministic --seed {seed} '
        compare_command += f'--attack-config-file {attack_config_file} '

        print(f'Compare | Running command\n{compare_command}')

        run_and_log(compare_command, compare_log_file)

        print('Compare | Finished')

if __name__ == '__main__':
    main()