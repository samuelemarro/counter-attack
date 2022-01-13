from pathlib import Path
import sys
sys.path.append('.')

import click

import utils

from adversarial_dataset import MergedDataset, MIPDataset

def parse_memory_log(log):
    parsed_log = {
        'main': {},
        'children' : {}
    }
    for line in log:
        elements = line.strip().split(' ')

        if elements[0] == 'CMDLINE':
            continue
        elif elements[0] == 'MEM':
            memory = float(elements[1])
            recording_timestamp = float(elements[2])
            parsed_log['main'][recording_timestamp] = memory
        elif elements[0] == 'CHLD':
            child_id = int(elements[1])
            memory = float(elements[2])
            recording_timestamp = float(elements[3])

            if child_id not in parsed_log['children']:
                parsed_log['children'][child_id] = {}
            parsed_log['children'][child_id][recording_timestamp] = memory
        else:
            raise RuntimeError(f'Found unknown command "{elements[0]}"')
    
    return parsed_log

def add_dataset(final_dataset : MergedDataset, dataset: MIPDataset, memory_log):
    assert dataset.stop == dataset.start + 1
    assert len(dataset.genuines) == 1

    if len(final_dataset.genuines) == 0:
        final_dataset.attack_configuration = dataset.attack_configuration
        final_dataset.misclassification_policy = dataset.misclassification_policy
        final_dataset.p = dataset.p

    index = dataset.start
    final_dataset.genuines[index] = dataset.genuines[0]
    final_dataset.labels[index] = dataset.labels[0]
    final_dataset.true_labels[index] = dataset.true_labels[0]
    final_dataset.adversarials[index] = dataset.adversarials[0]
    final_dataset.lower_bounds[index] = dataset.lower_bounds[0]
    final_dataset.upper_bounds[index] = dataset.upper_bounds[0]

    extra_info = dataset.extra_infos[0]
    for run_name in ['main', 'exploration']:
        if extra_info[run_name] is not None:
            for run in extra_info[run_name]:
                del run['logs']

    final_dataset.extra_infos[index] = extra_info
    final_dataset.global_extra_infos[index] = dataset.global_extra_info
    final_dataset.generation_kwargs[index] = dataset.generation_kwargs
    final_dataset.memory_logs[index] = memory_log

def read_datasets():
    for domain in ['mnist', 'cifar10']:
        for architecture in ['a', 'b', 'c']:
            for test_type in ['standard', 'adversarial', 'relu']:
                final_dataset_path = Path('final') / f'{domain}-{architecture}-{test_type}.zip'

                if not final_dataset_path.exists():
                    results_dir = Path('mip_results') / test_type / f'{domain}-{architecture}'
                    print('Checking', results_dir)

                    final_dataset = MergedDataset()

                    for dataset_path in results_dir.iterdir():
                        dataset_path = dataset_path.with_suffix('.zip')
                        dataset = utils.load_zip(dataset_path)

                        stem = dataset_path.stem

                        memory_log_path = Path('logs') / test_type / f'{domain}-{architecture}' / stem / 'mip_memory.dat'
                        if memory_log_path.exists():
                            with open(memory_log_path, 'r') as f:
                                memory_log = f.readlines()
                            memory_log = parse_memory_log(memory_log)
                        else:
                            memory_log = None

                        add_dataset(final_dataset, dataset, memory_log)
                    
                    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)

                    utils.save_zip(final_dataset, final_dataset_path)

@click.command()
def main():
    read_datasets()

if __name__ == '__main__':
    main()