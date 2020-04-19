import logging

import click

import commands

@click.group()
def main():
    pass

# TODO: Scaricare da un server con un nome porta a una violazione dell'anonimato?

# TODO: consistent_random deve anche prendere in considerazione il seed impostato
# TODO: Usare modelli pretrained o abbandonarli in favore di custom-trained per CIFAR10 e SVHN?

main.add_command(commands.accuracy)
main.add_command(commands.attack)
main.add_command(commands.attack_matrix)
main.add_command(commands.compare)
main.add_command(commands.cross_validation)
main.add_command(commands.distance_dataset)
main.add_command(commands.evasion)
main.add_command(commands.train_approximator)
main.add_command(commands.train_classifier)

if __name__  == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s: %(message)s')
    main()