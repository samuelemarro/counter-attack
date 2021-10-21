
start cmd.exe cmd /k "python batch_processing/train_and_prune.py cifar10 b2 standard --load-checkpoint"
start cmd.exe cmd /k "python batch_processing/train_and_prune.py cifar10 b2 adversarial --load-checkpoint"
start cmd.exe cmd /k "python batch_processing/train_and_prune.py cifar10 b2 relu --load-checkpoint"