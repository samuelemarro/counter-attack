"""Attack variants that return when an adversarial example within a certain distance is found.
"""

# fast_gradient does not require early rejection because it only has
# one step

from early_rejection_attacks.iterative_projected_gradient import L2ERPGDAttack, LinfERPGDAttack, L2ERBasicIterativeAttack, LinfERBasicIterativeAttack
from early_rejection_attacks.carlini_wagner import ERCarliniWagnerL2Attack, ERCarliniWagnerLinfAttack