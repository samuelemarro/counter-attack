import logging

import click

import commands

@click.group()
def main():
    pass

# TODO: Scaricare da un server con un nome porta a una violazione dell'anonimato?

# TODO: consistent_random deve anche prendere in considerazione il seed impostato
# TODO: Usare modelli pretrained o abbandonarli in favore di custom-trained per CIFAR10 e SVHN?
# Obiettivo: Arrivare a modelli Sequential MNIST, CIFAR10 e SVHN che possano essere attaccati da MIP
# Nota: MIP supporta le skip-connections
# TODO: Testare la correttezza di MIP
# TODO: Aggiungere codice che controlla la correttezza del modello caricato da MIP
# TODO: Abbandonare modelli pretrained, trasferirsi a modelli in-house?
# TODO: Se necessario, si può aggiungere l'addestramento ReLU stability
# TODO: Addestrare modelli in-house? Sì, per tutti e tre i dataset
# TODO: Implementare il test basato sulla distanza triangolare
# TODO: Data augmentation?
# TODO: Distinguere tra "classifier standard" e "classifier progettati per MIP"?
# TODO: Abbandonare sistema di pretrained, caricare i modelli e metterli in una cartella, con percorsi di default?
# TODO: get_dataset -> parse_dataset e simili
# TODO: Aggiungere l'opzione --architecture
# Questo mi permetterebbe di giustificare un'accuratezza più bassa

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
main.add_command(commands.perfect_approximation)
main.add_command(commands.train_approximator)
main.add_command(commands.train_classifier)

if __name__  == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s: %(message)s')
    main()