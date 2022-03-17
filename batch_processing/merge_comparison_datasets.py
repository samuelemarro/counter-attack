from pathlib import Path
import sys
sys.path.append('.')

import click

import utils

from adversarial_dataset import AttackComparisonDataset, MergedComparisonDataset


def add_dataset(final_dataset : MergedComparisonDataset, dataset: AttackComparisonDataset, log):
    assert dataset.stop == dataset.start + 1
    assert len(dataset.genuines) == 1

    if len(final_dataset.genuines) == 0:
        final_dataset.attack_configuration = dataset.attack_configuration
        final_dataset.attack_names = dataset.attack_names
        final_dataset.misclassification_policy = dataset.misclassification_policy
        final_dataset.p = dataset.p

    index = dataset.start
    final_dataset.genuines[index] = dataset.genuines[0]
    final_dataset.labels[index] = dataset.labels[0]
    final_dataset.true_labels[index] = dataset.true_labels[0]
    final_dataset.attack_results[index] = dataset.attack_results[0]
    final_dataset.generation_kwargs[index] = dataset.generation_kwargs
    final_dataset.logs[index] = log

def read_datasets(dataset_dir, output_dir, log_dir):
    for domain in ['mnist', 'cifar10']:
        for architecture in ['a', 'b', 'c']:
            for test_type in ['standard', 'adversarial', 'relu']:
                final_dataset_path = Path(output_dir) / f'{domain}-{architecture}-{test_type}.zip'

                if not final_dataset_path.exists():
                    results_dir = Path(dataset_dir) / test_type / f'{domain}-{architecture}'
                    print('Checking', results_dir)

                    final_dataset = MergedComparisonDataset()

                    for dataset_path in results_dir.iterdir():
                        dataset_path = dataset_path.with_suffix('.zip')
                        dataset = utils.load_zip(dataset_path)

                        stem = dataset_path.stem

                        log_path = Path(log_dir) / test_type / f'{domain}-{architecture}' / stem / 'compare.log'
                        if log_path.exists():
                            with open(log_path, 'r') as f:
                                log = f.readlines()
                        else:
                            log = None

                        add_dataset(final_dataset, dataset, log)
                    
                    final_dataset_path.parent.mkdir(parents=True, exist_ok=True)

                    utils.save_zip(final_dataset, final_dataset_path)

@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True))
@click.argument('log_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(dataset_dir, output_dir, log_dir):
    read_datasets(dataset_dir, output_dir, log_dir)

if __name__ == '__main__':
    main()