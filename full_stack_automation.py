import os

architecture = 'c'
domain = 'cifar10'

# TODO: 8/255 o 2/255? 0.3 o 0.1?

# Comuni
attack = 'pgd'
pre_attack = '[bim, brendel, carlini, pgd, uniform]'
rs_start_epoch_ratio = 0.8
data_augmentation = True
learning_rate = 1e-4
adversarial_eps_growth_start = 0.01
adversarial_eps_growth_epoch_ratio = 0.5
weight_pruning_threshold = 1e-3
relu_pruning_threshold = 0.9

if domain == 'cifar10':
    epochs = 250
    batch_size = 128 # Quella per naive IA, ma va bene per quella advanced

    adversarial_eps = 2/255
    l1_regularization = 1e-5
    rs_regularization = 1e-3
elif domain == 'mnist':
    epochs = 70
    batch_size = 32

    adversarial_eps = 0.1
    l1_regularization = 2e-5
    rs_regularization = 12e-5


# Derivati dagli altri parametri

rs_batch_size = batch_size
rs_start_epoch = int(epochs * rs_start_epoch_ratio) + 1
rs_eps = adversarial_eps # TODO: RS eps coincide con adversarial eps?
adversarial_eps_growth_epoch = int(epochs * adversarial_eps_growth_epoch_ratio)

if data_augmentation:
    data_augmentation_string = '--flip --translation 0.1 --rotation 15'
else:
    data_augmentation_string = ''

# TODO: La data augmentation influenza il valore dei parametri RS?

standard_state_dict = f'trained-models/robust-classifiers/{domain}-{architecture}-final.pth'
weight_pruned_state_dict = f'trained-models/weight-pruned-classifiers/{domain}-{architecture}-final-w.pth'
relu_pruned_state_dict = f'trained-models/relu-pruned-classifiers/{domain}-{architecture}-final-wr.pth'

state_dict_name = 'final-wr'
count = 20

if not os.path.exists(standard_state_dict):
    os.system(f'python cli.py train-classifier {domain} {architecture} std:train {epochs} {standard_state_dict} --batch-size {batch_size} --learning-rate {learning_rate} {data_augmentation_string} --l1-regularization {l1_regularization} --rs-regularization {rs_regularization} --rs-eps {rs_eps} --adversarial-training {attack} --adversarial-p linf --adversarial-ratio 1 --adversarial-eps {adversarial_eps} --rs-minibatch {rs_batch_size} --rs-start-epoch {rs_start_epoch} --adversarial-eps-growth-start {adversarial_eps_growth_start} --adversarial-eps-growth-epoch {adversarial_eps_growth_epoch}')
if not os.path.exists(weight_pruned_state_dict):
    os.system(f'python cli.py prune-weights {domain} {architecture} {standard_state_dict} {weight_pruned_state_dict} {weight_pruning_threshold}')
if not os.path.exists(relu_pruned_state_dict):
    os.system(f'python cli.py prune-relu {domain} {architecture} std:train {weight_pruned_state_dict} {relu_pruned_state_dict} {relu_pruning_threshold}')
os.system(f'python full_stack.py {domain} {architecture} {pre_attack} {count} --state-dict {relu_pruned_state_dict} --state-dict-name {state_dict_name}')