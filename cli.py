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
# TODO: Aggiungere codice che controlla la correttezza del modello caricato da MIP
# TODO: get_dataset -> parse_dataset e simili
# TODO: Eseguire gli addestramenti adversarial e RS
# TODO: Linf, L2 -> inf, 2
# TODO: Passare a formattazione nuova?

# TODO: Rimuovere sanity_test e svhn

# TODO: Quando non trova il sample di partenza (a causa di bound troppo stretti), fa 7200 secondi sprecati.
# Contemporaneamente, usare dei bound troppo larghi rallenta incredibilmente l'esecuzione
# Nota che il fatto che anche se la soluzione è infeasible non è detto che MIP non riesca a trovare uno start
# Avrei perciò un'idea: se ci fosse un modo per sapere molto velocemente se MIP riesce a trovare uno start, potrei usare
# un bound minuscolo e crescere da lì
# Problema #2: nella maggior parte dei casi, sembra che una volta trovato il valore giusto, faccia comunque pena.

# TODO: Si può migliorare l'inizializzazione delle variabili in mip_interface.jl? Non sono sicuro che abbia eliminato tutte
# le variabili
# TODO: Devo abbandonare l'idea di sapere subito se un tempo è feasible? Se è così, devo permettere di passare un tempo diverso di esplorazione


# TODO: Qual è l'adversarial ratio?
# TODO: --adversarial-eps-growth-start = 1 dovrebbe non fare nessuna differenza, ma non sono sicuro
# TODO: Dare un nome di diverso a --adversarial-growth-eps-start?

# TODO: Aggiungere un valore di default di eps che causi errore se non viene overridato

# TODO: Tanto logging

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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(name)s:%(levelname)s: %(message)s')
    main()
