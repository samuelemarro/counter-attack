import click
import os

# Parametri originali Madry per CIFAR10 (che usava range 0-255): step size 2, 10 step
# Però in teoria il gradiente è già scalato

@click.command()
@click.argument('domain', type=click.Choice(['cifar10', 'mnist']))
@click.argument('architecture', type=click.Choice(['a', 'b', 'c']))
@click.argument('test_name', type=click.Choice(['standard', 'adversarial', 'relu']))
def main(domain, architecture, test_name):
    # Common hyperparameters
    training_attack = 'pgd'
    learning_rate = 1e-4
    seed = 0
    epochs = 425

    # Madry replaces all genuines with adversarials
    adversarial_ratio = 1

    weight_pruning_threshold = 1e-3
    relu_pruning_threshold = 0.9
    adversarial_p = 'linf'
    attack_config_file = 'default_attack_configuration.cfg'

    if domain == 'cifar10':
        # We use Xiao's eps=2/255 hyperparameter set

        # Quella per naive IA, ma va bene per quella advanced
        batch_size = 128

        adversarial_eps = 2/255
        l1_regularization = 1e-5
        rs_regularization = 1e-3
    elif domain == 'mnist':
        # We use Xiao's eps=0.1 hyperparameter set
        batch_size = 32

        adversarial_eps = 0.1
        l1_regularization = 2e-5
        rs_regularization = 12e-5
    else:
        raise RuntimeError()

    # Derived hyperparameters
    rs_minibatch_size = batch_size

    # rs_eps is equal to adversarial_eps
    rs_eps = adversarial_eps

    checkpoint_every = int(epochs / 20)

    if test_name == 'relu':
        standard_state_dict = f'trained-models/classifiers/{test_name}/standard/{domain}-{architecture}.pth'
    else:
        standard_state_dict = f'trained-models/classifiers/{test_name}/{domain}-{architecture}.pth'

    if os.path.exists(standard_state_dict):
        print('Skipping Training')
    else:
        training_command = f'python cli.py train-classifier {domain} {architecture} std:train {epochs} {standard_state_dict} '
        training_command += f'--batch-size {batch_size} --learning-rate {learning_rate} --checkpoint-every {checkpoint_every} '

        # Add data augmentation for all tests
        translation = 0.1
        rotation = 15
        training_command += f'--flip --translation {translation} --rotation {rotation} '

        # Add determinism
        training_command += f'--deterministic --seed {seed} '

        if test_name == 'adversarial' or test_name == 'relu':
            training_command += f'--adversarial-training {training_attack} --adversarial-p {adversarial_p} --adversarial-ratio {adversarial_ratio} '
            training_command += f'--adversarial-eps {adversarial_eps} '
            training_command += f'--attack-config-file {attack_config_file} '

        if test_name == 'relu':
            training_command += f'--l1-regularization {l1_regularization} --rs-regularization {rs_regularization} --rs-eps {rs_eps} '
            training_command += f'--rs-minibatch {rs_minibatch_size} '

        print(f'Training | Running command\n"{training_command}"')
        os.system(training_command)

    if test_name == 'relu':
        weight_pruned_state_dict = f'trained-models/{test_name}/weight-pruned/{domain}-{architecture}.pth'
        relu_pruned_state_dict = f'trained-models/{test_name}/relu-pruned/{domain}-{architecture}.pth'

        if os.path.exists(weight_pruned_state_dict):
            print('Skipping Weight Pruning')
        else:
            weight_pruning_command = f'python cli.py prune-weights {domain} {architecture} {standard_state_dict} '
            weight_pruning_command += f'{weight_pruned_state_dict} {weight_pruning_threshold} '
            weight_pruning_command += '--deterministic '
            print(f'Weight Pruning | Running command\n"{weight_pruning_command}".')
            os.system(weight_pruning_command)

        if os.path.exists(relu_pruned_state_dict):
            print('Skipping ReLU Pruning')
        else:
            relu_pruning_command = f'python cli.py prune-relu {domain} {architecture} std:train '
            relu_pruning_command += f'{training_attack} {adversarial_p} {adversarial_ratio} {adversarial_eps} '
            relu_pruning_command += f'{weight_pruned_state_dict} {relu_pruning_threshold} {relu_pruned_state_dict} '
            relu_pruning_command += f'--batch-size {batch_size} '
            relu_pruning_command += f'--attack-config-file {attack_config_file} '
            relu_pruning_command += f'--deterministic --seed {seed}'

            print(f'ReLU Pruning | Running command\n"{relu_pruning_command}".')
            os.system(relu_pruning_command)

if __name__ == '__main__':
    main()