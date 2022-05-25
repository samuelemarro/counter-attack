import sys

sys.path.append('.')

import click
import matplotlib.pyplot as plt

import utils

@click.command()
@click.argument('path')
@click.argument('threshold', type=float)
def main(path, threshold):
    dataset = utils.load_zip(path)

    bounds = list(zip(dataset.lower_bounds.values(), dataset.upper_bounds.values()))
    print('Total bounds:', len(bounds))
    
    bounds = [bound for bound in bounds if bound[0] is not None and bound[1] is not None]
    print('Non-None bounds:', len(bounds))

    bounds = [bound for bound in bounds if bound[0] < 1e40 and bound[1] < 1e40]
    print('Non-inf bounds:', len(bounds))

    valid_bounds = [bound for bound in bounds if bound[1] - bound[0] < threshold]
    print('Valid bounds:', len(valid_bounds))

    plt.hist([bound[1] - bound[0] for bound in bounds], bins=100)

    plt.show()

if __name__ == '__main__':
    main()