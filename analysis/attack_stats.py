import sys

sys.path.append('.')

import click

import adversarial_dataset
import utils


@click.command()
@click.argument('domain')
@click.argument('architecture')
@click.argument('test')
@click.option('--median-average-atol', type=float, default=adversarial_dataset.MEDIAN_AVERAGE_ATOL)
@click.option('--attack-ranking-atol', type=float, default=adversarial_dataset.DISTANCE_ATOL)
@click.option('--pairwise-comparison-atol', type=float, default=adversarial_dataset.DISTANCE_ATOL)
@click.option('--win-rate-atol', type=float, default=adversarial_dataset.DISTANCE_ATOL)
def main(domain, architecture, test, median_average_atol, attack_ranking_atol, pairwise_comparison_atol, win_rate_atol):
    print('=' * 80)
    print(domain, architecture, test)
    print('=' * 80)

    comparison_path = f'final-comparison/{domain}-{architecture}-{test}.zip'
    print('Loading comparison dataset...')
    comparison_dataset = utils.load_zip(comparison_path)
    print('Done.')

    comparison_dataset.print_stats(
        median_average_atol=median_average_atol,
        attack_ranking_atol=attack_ranking_atol,
        pairwise_comparison_atol=pairwise_comparison_atol,
        win_rate_atol=win_rate_atol
    )

    print('\n' * 3)

if __name__ == '__main__':
    main()
