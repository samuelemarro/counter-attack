import numpy as np

import utils

tolerance_dataset_path = 'mnist-mini-100-mip-blind.zip'
no_tolerance_dataset_path = 'mnist-mini-100-mip-blind-zero-tolerance.zip'

average_l1_differences = []
linf_differences = []

tolerance_dataset = utils.load_zip(tolerance_dataset_path)
no_tolerance_dataset = utils.load_zip(no_tolerance_dataset_path)

for (genuine_tolerance, adversarial_tolerance), (genuine_no_tolerance, adversarial_no_tolerance) in zip(tolerance_dataset.to_distance_dataset(None), no_tolerance_dataset.to_distance_dataset(None)):
    adversarial_tolerance = adversarial_tolerance.numpy()
    adversarial_no_tolerance = adversarial_no_tolerance.numpy()
    average_l1 = np.average(np.abs(adversarial_tolerance - adversarial_no_tolerance))
    linf = np.max(np.abs(adversarial_tolerance - adversarial_no_tolerance))

    average_l1_differences.append(average_l1)
    linf_differences.append(linf)

print(np.average(average_l1_differences))
print(np.max(average_l1_differences))
print(np.average(linf_differences))
print(np.max(linf_differences))