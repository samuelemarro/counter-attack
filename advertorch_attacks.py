import advertorch


class CarliniL2DetectorLoss(advertorch.attacks.CarliniWagnerL2Attack):
    def __init__(self, predict, num_classes, confidence=0,
                 targeted=False, learning_rate=0.01,
                 binary_search_steps=9, max_iterations=10000,
                 abort_early=True, initial_const=1e-3,
                 clip_min=0., clip_max=1., loss_fn=None):
        super().__init__(predict, num_classes,
        confidence, targeted, learning_rate, binary_search_steps,
        max_iterations, abort_early, initial_const, clip_min, clip_max, loss_fn)

    