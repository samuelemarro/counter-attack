# PyTorch has some serious bugs concerning dll loading:
# If PyTorch is loaded before Julia, Julia's import fails.
# We therefore import Julia before anything else
import julia
from julia import Base

import logging

import click

import commands

@click.group()
def main():
    pass

# TODO: consistent_random deve anche prendere in considerazione il seed impostato
# Obiettivo: Arrivare a modelli Sequential MNIST, CIFAR10 e SVHN che possano essere attaccati da MIP
# Nota: MIP supporta le skip-connections
# TODO: Aggiungere codice che controlla la correttezza del modello caricato da MIP
# TODO: get_dataset -> parse_dataset e simili
# TODO: Nell'adversarial training, non si ha modo di dire all'attacco di usare --adversarial-eps
#       O si toglie l'opzione, o, devo trovare un modo per permettere l'override
#       In alternativa si può lasciare così dove l'attacco è best-effort ma si rifiutano quelli oltre --adversarial-eps 

# TODO: Eseguire gli addestramenti adversarial e RS
# TODO: Aggiungere start e stop ai parametri degli adversarial dataset
# TODO: Linf, L2 -> inf, 2
# TODO: Passare a formattazione nuova?

# TODO: Rimuovere sanity_test e svhn

main.add_command(commands.accuracy)
main.add_command(commands.attack)
main.add_command(commands.attack_matrix)
main.add_command(commands.compare)
main.add_command(commands.cross_validation)
main.add_command(commands.distance_dataset)
main.add_command(commands.evasion)
main.add_command(commands.mip)
main.add_command(commands.perfect_approximation)
main.add_command(commands.prune_relu)
main.add_command(commands.prune_weights)
main.add_command(commands.train_approximator)
main.add_command(commands.train_classifier)
main.add_command(commands.tune_mip)

if __name__  == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s: %(message)s')
    main()