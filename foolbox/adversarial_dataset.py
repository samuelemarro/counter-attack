class AdversarialDataset:
    def __init__(self, genuines, adversarials):
        assert len(genuines) == len(adversarials)

        self.genuines = genuines
        self.adversarials = adversarials

    def __iter__(self):
        return zip(self.genuines, self.adversarials).__iter__()

class AdversarialDistanceDataset:
    def __init__(self, attack_name, images, distances):
        assert len(images) == len(distances)
        
        self.attack_name = attack_name
        self.images = images
        self.distances = distances

    def __iter__(self):
        return zip(self.images, self.distances).__iter__()