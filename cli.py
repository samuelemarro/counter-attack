import logging

import click

import commands

@click.group()
def main():
    pass

# TODO: Scaricare da un server con un nome porta a una violazione dell'anonimato?

# TODO: consistent_random deve anche prendere in considerazione il seed impostato
# Obiettivo: Arrivare a modelli Sequential MNIST, CIFAR10 e SVHN che possano essere attaccati da MIP
# Nota: MIP supporta le skip-connections
# TODO: Aggiungere codice che controlla la correttezza del modello caricato da MIP
# TODO: Se necessario, si può aggiungere l'addestramento ReLU stability
# TODO: Implementare il test basato sulla distanza triangolare
# TODO: Abbandonare sistema di pretrained, caricare i modelli e metterli in una cartella, con percorsi di default?
# TODO: get_dataset -> parse_dataset e simili
# TODO: Fix fast_boolean_choice
# TODO: Brendel & Bethge (di foolbox)
# TODO: Passare a foolbox?

# TODO: Finire RS_Loss & simili
# TODO: Eseguire gli addestramenti adversarial e RS

# TODO: Esiste un metodo per rendere gli attacchi più efficaci? Questo potrebbe essere l'anello mancante
# per concludere il ragionamento empirico (potrei confrontare efficacia prima e dopo)
# In altre parole, serve qualcosa che rende il classifier provably più robusto (o ugualmente robusto),
# ma gli attacchi ottengono una distortion più vicina all'ottimale

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