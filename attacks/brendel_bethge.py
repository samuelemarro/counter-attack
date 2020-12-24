import foolbox


class BrendelBethge:
    def __init__(self, overshoot=1.1, steps=1000, lr=1e-3, lr_decay=0.5, lr_num_decay=20, momentum=0.8, binary_search_steps=10):
        inner_attack = foolbox.attacks.brendel_bethge.LinfinityBrendelBethgeAttack(overshoot=overshoot,
                        steps=steps, lr=lr, lr_decay=lr_decay, lr_num_decay=lr_num_decay,
                        momentum=momentum, binary_search_steps=binary_search_steps)
        # TODO: Finire