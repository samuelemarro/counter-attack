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
# TODO: Eseguire gli addestramenti adversarial e RS
# TODO: Linf, L2 -> inf, 2

# TODO: Rimuovere sanity_test e svhn
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
# TODO: Il sistema di return_best è re-implementato in epsilon_binary_search e uniform_noise. Toglierlo?
# TODO: Finire il debugging di attacks/mip.py e uniform.py
# TODO: Il comando mip è un comando a sé da debuggare!

# TODO: Ri-eseguire l'addestramento? Guardo quale aveva vinto per ogni categoria e prendo quello

# TODO: Rimuovere architetture inutili
# TODO: Eliminare svhn.py
# TODO: Rimuovere consistent_random
# TODO: Eliminare sanity_tests.py
# TODO: accuracy.py che fa l'override di dataset per --from-adversarial-dataset non è bellissimo
# TODO: LA NORMALIZZAZIONE DI CIFAAAAAAAARRRRRRRRR
# TODO: è giusto che si usi l'inizializzazione intelligente per Brendel? Secondo me no
# TODO: Supporto salvataggio checkpoint

"""
Lista dei moduli ancora da controllare

- attacks
    - mip.py
    - uniform_noise.py
- commands
    - accuracy.py (quasi fatto)
    - attack_matrix.py [p]
    - attack.py
    - compare.py
    - cross_validation.py
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
    - mnist.py
- adversarial_dataset.py
- detectors.py [p]
- full_stack_automation.py
- full_stack.py
- mip_interface.jl
- parsing.py (manca solo la normalizzazione di CIFAR)
- torch_utils.py
- training.py
- utils.py

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
