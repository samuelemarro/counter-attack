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
# TODO: Verificare che attack_comparison_test funzioni ancora

# TODO: Togliere il boolean indexing da uniform_noise.py
# TODO: Finire il debugging di attacks/mip.py
# TODO: Il comando mip è un comando a sé da debuggare!

# TODO: Ri-eseguire l'addestramento? Guardo quale aveva vinto per ogni categoria e prendo quello

# TODO: accuracy.py che fa l'override di dataset per --from-adversarial-dataset non è bellissimo
# TODO: Droppare completamente il supporto per L2?
# TODO: Il seeding dovrebbe anche impostare np.random
# TODO: Parametri dei vari attacchi
# TODO: Scegliere una misclassification_policy


# TODO: Cercare i valori corretti di l1 & co. tramite line search?
# Appunto: CIFAR10-A ha avuto out-of-RAM ed è stato riavviato a partire da 85. è indietro di 35-40 epochs.
# TODO: Passare a 1-indexing per le epochs nel salvataggio checkpoint?

"""
Lista dei moduli ancora da controllare

- attacks
    - carlini_wagner
    - kbesttarget
    - mip.py
    - random_target
- commands
    - accuracy.py (quasi fatto)
    - attack_matrix.py [p]
    - attack.py (mancano AttackPool e attack_test)
    - compare.py
    - cross_validation.py [p]
    - distance_dataset.py
    - evasion.py [p]
    - mip.py
    - perfect_approximation.py [p]
    - prune_relu.py
    - prune_weights.py
    - train_approximator.py
    - train_classifier.py
    - tune_mip.py [p?]
- models
    - cifar.py
- adversarial_dataset.py
- detectors.py [p]
- full_stack_automation.py
- full_stack.py
- mip_interface.jl
- parsing.py (manca solo AttackPool, detectors e validazione)
- training.py (in teoria avevo già fatto una prima passata)
- utils.py
    Fatti:
    - save_zip
    - load_zip
    - show_images
    - maybe_stack
    - powerset
- default_attack_configuration.cfg

Paths:
- train_classifier (quasi fatto)
- prune_relu
- prune_weights
- attack (75%)
- compare
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
