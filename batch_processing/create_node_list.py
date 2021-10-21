from pathlib import Path

import click

@click.command()
@click.argument('all_nodes_path', type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument('chosen_nodes_path', type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument('excluded_nodes_path', type=click.Path(dir_okay=False, file_okay=True))
def main(all_nodes_path, chosen_nodes_path, excluded_nodes_path):
    with open(all_nodes_path) as f:
        all_nodes = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
    with open(chosen_nodes_path) as f:
        chosen_nodes = [l.strip() for l in f.readlines() if len(l.strip()) > 0]

    assert all(node in all_nodes for node in chosen_nodes)
    
    excluded_nodes = [node for node in all_nodes if node not in chosen_nodes]

    assert all(node in excluded_nodes or node in chosen_nodes for node in all_nodes)
    
    with open(excluded_nodes_path, 'w') as f:
        f.write(','.join(excluded_nodes))

if __name__ == '__main__':
    main()