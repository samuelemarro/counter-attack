import sys

sys.path.append('.')

import click

import utils


@click.command()
@click.argument('domain')
@click.argument('architecture')
@click.argument('test')
def main(domain, architecture, test):
    print('=' * 80)
    print(domain, architecture, test)
    print('=' * 80)

    comparison_path = f'final-comparison/{domain}-{architecture}-{test}.zip'
    print('Loading comparison dataset...')
    comparison_dataset = utils.load_zip(comparison_path)
    print('Done.')

    comparison_dataset.print_stats()

    print('\n' * 3)

if __name__ == '__main__':
    main()
