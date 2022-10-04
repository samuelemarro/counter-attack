from random import shuffle
import sys
sys.path.append('.')

import json
import click

import parsing
import utils

@click.command()
@click.argument('domain')
def main(domain):
    utils.set_seed(0)

    distances_path = f'analysis/distances/mip/{domain}-a-standard.json'
    with open(distances_path) as f:
        distances = json.load(f)

    indices = [int(k) for k in distances.keys()]

    dataset = parsing.parse_dataset(domain, 'std:test')

    bins = [list() for _ in range(10)]

    for index in indices:
        label = dataset[index][1]
        bins[label].append(index)

    for bin in bins:
        shuffle(bin)
    # bins = [shuffle(bin) for bin in bins]
    print(bins[0])

    final_list = []

    for i in range(max([len(bin) for bin in bins])):
        bin_id_list = list(range(10))
        shuffle(bin_id_list)
        for bin_id in bin_id_list:
            if i < len(bins[bin_id]):
                final_list.append(bins[bin_id][i])

    with open(f'fooling/sorted_indices_{domain}.json', 'w') as f:
        json.dump(final_list, f)

if __name__ == '__main__':
    main()