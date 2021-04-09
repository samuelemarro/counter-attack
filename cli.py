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

# TODO: Eseguire gli addestramenti adversarial e RS
# TODO: Linf, L2 -> inf, 2

# TODO: Rimuovere gli attacchi non usati

# Nota: Se negli evasion l'originale viene rifiutato ma l'adversarial no, l'adversarial conta
# come successo anche se ha mantenuto la stessa label di partenza
# TODO: Testare!

# TODO: Quando non trova il sample di partenza (a causa di bound troppo stretti), fa 7200 secondi sprecati.
# Contemporaneamente, usare dei bound troppo larghi rallenta incredibilmente l'esecuzione
# Nota che il fatto che anche se la soluzione è infeasible non è detto che MIP non riesca a trovare uno start
# Avrei perciò un'idea: se ci fosse un modo per sapere molto velocemente se MIP riesce a trovare uno start, potrei usare
# un bound minuscolo e crescere da lì
# Problema #2: nella maggior parte dei casi, sembra che una volta trovato il valore giusto, faccia comunque pena.

# TODO: Devo abbandonare l'idea di sapere subito se un tempo è feasible? Se è così, devo permettere di passare un tempo diverso di esplorazione

# TODO: --adversarial-eps-growth-start = 1 dovrebbe non fare nessuna differenza, ma non sono sicuro
# TODO: Dare un nome di diverso a --adversarial-growth-eps-start?

# TODO: Aggiungere un valore di default di eps che causi errore se non viene overridato

# TODO: Tanto logging

# TODO: Formalizzare full_stack.py
# TODO: LABEL DEI MISCLASSIFIED (SOPRATTUTTO COME GESTIRLE NELL'ADV.TRAINING) (Nota: nell'adversarial training non dà problemi in teoria, check)

# TODO: Finire il debugging di attacks/mip.py
# TODO: Il comando mip è un comando a sé da debuggare!

# TODO: accuracy.py che fa l'override di dataset per --from-adversarial-dataset non è bellissimo
# TODO: Droppare completamente il supporto per L2?
# TODO: Parametri dei vari attacchi
# TODO: Scegliere una misclassification_policy


# TODO: Cercare i valori corretti di l1 & co. tramite line search?
# Appunto: CIFAR10-A ha avuto out-of-RAM ed è stato riavviato a partire da 85. è indietro di 35-40 epochs.
# TODO: Passare a 1-indexing per le epochs nel salvataggio checkpoint?
# TODO: Evitare duplicazione del codice in train/val?
# TODO: topk e random_target hanno delle sincronizzazioni
# TODO: Carlini Linf usa parametri molto più tranquilli nell'implementazione originale
# TODO: parsing non ha modo di sapere qual è il device corretto per Carlini Linf
# TODO: Ci sono pezzi del codice che danno per scontato che le immagini siano 4D? Non è un problema, anche MNIST ha una batch dim
# TODO: torch.max non ha il comportamento atteso?
# TODO: Verificare che il nuovo .cfg abbia senso
# TODO: np.seed è deprecato
# TODO: La misclassification policy "remove" deve restituirli come None?
# TODO [p]: Debuggare il comportamento targeted
# Appunto: PGD è abbastanza decente con i parametri da training

# TODO: ADVERSARIAL TRAINING NON USA IL DATA AUGMENTATION

# TODO: Il preprocessor si basa sulla media di tutti i sample del training set, nonostante parte di essi vengano usati poi
# per il validation set. Questo non è particolarmente grave, ma è qualcosa su cui riflettere

# TODO: Breaking bug: training.py caricava a ogni epoch il best model [fixed]. Rifare addestramenti standard
# TODO: Breaking bug: relu_stable calcolava RS sui clean, non sugli adversarials [fixed, da debuggare]
# TODO: Breaking bug: relu_stable usa xent media, non sommata (mentre usa la sommata per l1?) [fixed, da debuggare] -> Anche io, a quanto pare
# TODO: Breaking bug: np.random.choice può estrarre più volte la stessa cosa [fixed]
# Vista la grande quantità di breaking bugs, si consiglia un confronto completo con relu_stable
# TODO: Breaking bug: L1 viene calcolata e scalata in maniera diversa [fixed, da debuggare]
# TODO: Breaking bug: ReLU Pruning viene fatto in maniera diversa. Confrontare quello e weight pruning con l'originale
# TODO: Perché l'implementazione originale ha anche un modello masked?

# TODO: Aggiungere --keep-best a training?

# TODO: Nell'originale di Xiao e Madry non usano manco la normalisation

# Appunto: conv_to_matrix preserva il grafo dei gradienti e fare la l1 sul linear è equivalente a fare una l1 sulla conv e
# moltiplicare per la dimensione (senza channel) dell'output. Nota però che linearized_model non è
# un leaf node, quindi non ha un grad

# Appunto: è giusto che se calcolo con la weight matrix ottengo un gradiente uguale a 49 volte il gradiente di conv?
# Sì, perché supponendo di avere il gradiente di una singola applicazione di conv, essa corrisponde a selezionare una zona, prendere
# input_channels canali, moltiplicarli per la convoluzione e ottenere un pixel con output_channels canali. Ciò viene ripetuto per ogni
# pixel dell'output, ovvero 7x7
# La versione matriciale semplicemente esplicita questa ripetizione, ottenendo una matrice che associa tutto l'input a tutto l'output, ma dove
# ogni pixel di output è influenzato esclusivamente dalla regione che avrebbe considerato la convoluzione.

# TODO: Tecnicamente il checkpoint dovrebbe salvare lo stato attuale di random, ma anche se non è diverso non dovrebbe essere grave
# TODO: Aggiungere supporto per il --choose-best.

"""
Lista dei moduli ancora da controllare

- attacks
    - k_best_target [p]
    - mip.py
    - random_target [p]
- commands
    - accuracy.py (quasi fatto)
    - attack_matrix.py [p]
    - cross_validation.py [p]
    - distance_dataset.py [p]
    - evasion.py [p]
    - mip.py
    - perfect_approximation.py [p]
    - prune_relu.py [da fare, ha un breaking]
    - prune_weights.py
    - train_approximator.py [p, manca supporto --choose-best]
    - train_classifier.py [manca supporto --choose-best]
    - tune_mip.py [p?]
- models
    - cifar.py
- adversarial_dataset.py
    Fatti:
    - AttackDataset (manca la conversione in AdversarialTrainingDataset)
    - AttackComparisonDataset
- detectors.py [p]
- full_stack_automation.py [?]
- full_stack.py [?]
- mip_interface.jl
- parsing.py (manca solo detectors [p])
    Controllati:
    - parse_dataset
    - parse_optimiser
    - parse_attack
    - validate_lp_distance
    - ParameterList
- tests.py
    Fatti:
    - attack
    - multiple_attack
- training.py (in teoria avevo già fatto una prima passata)
    Fatti:
    - split_dataset
    - IndexedDataset
    - l1_loss
    - train [mancano i TODO, pulire i print e un po' di test generali]
    - adversarial_training
    - model_to_linear_sequence [mancano i TODO]
    - EarlyStop [in corso]
- utils.py
    Fatti:
    - save_zip
    - load_zip
    - show_images
    - maybe_stack
    - powerset
    - read_attack_config_file
    - clip_adversarial
    - create_label_dataset
    - get_labels
    - one_many_adversarial_distance
    - adversarial_distance
    - AttackConfig
    - remove_failed
    - misclassified
    - misclassified_outputs
- default_attack_configuration.cfg

Paths:
- train_classifier (quasi fatto)
- prune_relu
- prune_weights
- attack (100%)
- compare (100%)
- mip
- perfect_approximation

Da implementare:
- gestione batching
- compattamento output

"""


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
