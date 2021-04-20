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

# TODO: Tanto logging

# TODO: Formalizzare full_stack.py

# TODO: Finire il debugging di attacks/mip.py
# TODO: Il comando mip è un comando a sé da debuggare!

# TODO: accuracy.py che fa l'override di dataset per --from-adversarial-dataset non è bellissimo
# TODO: Droppare completamente il supporto per L2?
# TODO: Parametri dei vari attacchi
# TODO: Scegliere una misclassification_policy


# TODO: Cercare i valori corretti di l1 & co. tramite line search?
# TODO: topk e random_target hanno delle sincronizzazioni
# TODO: Carlini Linf usa parametri molto più tranquilli nell'implementazione originale
# TODO: np.seed è deprecato
# TODO: La misclassification policy "remove" deve restituirli come None?
# TODO [p]: Debuggare il comportamento targeted
# Appunto: PGD è abbastanza decente con i parametri da training

# TODO: ADVERSARIAL TRAINING NON USA IL DATA AUGMENTATION. Ma nel paper originale di Madry sì

# TODO: Il preprocessor si basa sulla media di tutti i sample del training set, nonostante parte di essi vengano usati poi
# per il validation set. Questo non è particolarmente grave, ma è qualcosa su cui riflettere

# TODO: Valori corretti delle data augmentations
# TODO: Rifare tutti gli addestramenti

# TODO: Nell'originale di Xiao e Madry non usano manco la normalisation
# TODO: Nell'originale hanno un ReLU al fondo (devo anche aggiungerlo io a Wong?) | NON È VERO!

# Appunto: conv_to_matrix preserva il grafo dei gradienti e fare la l1 sul linear è equivalente a fare una l1 sulla conv e
# moltiplicare per la dimensione (senza channel) dell'output. Nota però che linearized_model non è
# un leaf node, quindi non ha un grad

# Appunto: è giusto che se calcolo con la weight matrix ottengo un gradiente uguale a 49 volte il gradiente di conv?
# Sì, perché supponendo di avere il gradiente di una singola applicazione di conv, essa corrisponde a selezionare una zona, prendere
# input_channels canali, moltiplicarli per la convoluzione e ottenere un pixel con output_channels canali. Ciò viene ripetuto per ogni
# pixel dell'output, ovvero 7x7
# La versione matriciale semplicemente esplicita questa ripetizione, ottenendo una matrice che associa tutto l'input a tutto l'output, ma dove
# ogni pixel di output è influenzato esclusivamente dalla regione che avrebbe considerato la convoluzione.

# TODO: GLI ADDESTRAMENTI VANNO FATTI CON CHOOSE-BEST E DETERMINISTIC
# TODO: TUTTO VA FATTO CON DETERMINISTIC E SEED
# TODO: I risultati dell'attacco vanno salvati!

# TODO: Se uso MaskedReLU in un modello obiettivo di attacco non-MIP, devo debuggare questo caso

# TODO: Se metto la data augmentation per RS training, bisogna anche usarla per prune-relu?

# Appunto: Se uso un dataloader multi-worker con delle data augmentations, c'è un bug molto comune
# https://www.reddit.com/r/MachineLearning/comments/mocpgj/p_using_pytorch_numpy_a_bug_that_plagues/

# TODO: confrontare tempi carlini mio vs tranquillo [ENTRO GIOVEDI]

# TODO: Togliere gli "Is this intentional?"

# TODO: Se cli.py non trova julia, deve fallire silenziosamente

# TODO: click.IntRange(1) => click.IntRange(1, None)
# TODO: Controllare .eval() nei vari comandi
# TODO: I tipi di attacco (standard, evasion...) dovrebbero essere in una sorta di enum?
# TODO: .cpu().detach() -> .detach().cpu()
# Idem per misclassification_policy

"""
Lista dei moduli ancora da controllare

- attacks
    - k_best_target [p]
    - mip.py
        Fatti:
        - __init__
        - _check_model
        - _run_mipverify
        - _mip_success
        - _find_perturbation_size
        - _mip_attack
    - random_target [p]
- commands
    - accuracy.py (quasi fatto)
    - attack_matrix.py [p]
    - cross_validation.py [p]
    - distance_dataset.py [p]
    - evasion.py [p]
    - perfect_approximation.py [p]
    - train_approximator.py [p]
    - tune_mip.py [p?]
- models
    - cifar.py
- adversarial_dataset.py
    Fatti:
    - AttackDataset (manca la conversione in AdversarialTrainingDataset)
    - AttackComparisonDataset
    - MIPDataset [mancano i test generali]
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
    - mip_test [mancano i test generali]
- training.py
    Fatti:
    - split_dataset
    - IndexedDataset
    - StartStopDataset
    - l1_loss
    - train
    - adversarial_training
    - model_to_linear_sequence
    - EarlyStop
    - ValidationTracker
    - conv_to_matrix
    - rs_loss
    - _interval_arithmetic
    - _interval_arithmetic_batch
    - _interval_arithmetic_all_batch
    - _compute_bounds_n_layers
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
    - set_seed
    - get_rng_state
    - set_rng_state
    - enable_determinism
- default_attack_configuration.cfg

Paths:
- train_classifier (100%)
- prune_relu (100%)
- prune_weights (100%)
- attack (100%)
- compare (100%)
- mip
- perfect_approximation

Da implementare:
- gestione batching
- compattamento output

"""

# Cosa guardare nei test generali?
# - Pooling nell'AdversarialDataset
# - Funzionamento attacchi e attack_test


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
