import click
import os

# TODO: Parametri PGD, al massimo li prendo da Madry
# TODO: #epochs per standard
# TODO: Dare anche una seconda occhiata a tutti gli output (magari una volta che eseguo proprio)

# Il range ha un +1, ma ci sono 10k epochs. WTF?
# Comunque fa ii/max_num_training_steps, dove max_num è 10k e ii cicla fino a max_num incluso
# Conclusione: se l'indice è effettivamente 0-indexed, fa 50 epochs dove current_eps è < eps (quindi si inizia alla 51 1-indexed)
# Se l'indice è 1-indexed, fa 49 epochs dove current_eps è < eps (quindi si inizia alla 50 1-indexed)

# Se guardi che il range ha +1, allora l'indice è 1-indexed
# Se guardi che passi ii, allora l'indice è 0-indexed

# Se imposto 100 epochs, lui farà 50 epochs con minore [0-49] e 51 con uguale [50-100]
# Seguendo questo principio, ha quindi senso

@click.command()
@click.argument('domain', type=click.Choice(['cifar10', 'mnist']))
@click.argument('architecture', type=click.Choice(['a', 'b', 'c']))
@click.argument('test_name', type=click.Choice(['standard', 'adversarial', 'relu']))
def main(domain, architecture, test_name):
    # Common hyperparameters
    training_attack = 'pgd'
    rs_start_epoch_ratio = 0.8
    learning_rate = 1e-4
    adversarial_eps_growth_start = 0.01
    adversarial_eps_growth_epoch_ratio = 0.5
    seed = 0

    # Madry replaces all genuines with adversarials
    adversarial_ratio = 1

    weight_pruning_threshold = 1e-3
    relu_pruning_threshold = 0.9
    adversarial_p = 'linf'

    if domain == 'cifar10':
        # We use Xiao's eps=8/255 hyperparameter set
        epochs = 250

        # Quella per naive IA, ma va bene per quella advanced
        batch_size = 128

        adversarial_eps = 8/255
        l1_regularization = 1e-5
        rs_regularization = 2e-3
    elif domain == 'mnist':
        # We use Xiao's eps=0.1 hyperparameter set
        epochs = 70
        batch_size = 32

        adversarial_eps = 0.1
        l1_regularization = 2e-5
        rs_regularization = 12e-5
    else:
        raise RuntimeError()

    # Derived hyperparameters
    rs_minibatch_size = batch_size
    rs_start_epoch = int(epochs * rs_start_epoch_ratio) + 1

    # rs_eps is equal to adversarial_eps
    rs_eps = adversarial_eps

    # If num_epochs=100, Xiao's original implementation uses a below-eps
    # value for the 50 epochs and eps for 51 epochs. Since that doesn't add up to
    # 100, we set rs_start_epoch so that 50 epochs use a below-eps value and 50 use eps
    adversarial_eps_growth_epoch = int(epochs * adversarial_eps_growth_epoch_ratio) + 1
    checkpoint_every = int(epochs / 20)

    if test_name == 'relu':
        standard_state_dict = f'trained-models/classifiers/{test_name}/standard/{domain}-{architecture}.pth'
    else:
        standard_state_dict = f'trained-models/classifiers/{test_name}/{domain}-{architecture}.pth'

    weight_pruned_state_dict = f'trained-models/{test_name}/weight-pruned/{domain}-{architecture}.pth'
    relu_pruned_state_dict = f'trained-models/{test_name}/relu-pruned/{domain}-{architecture}.pth'

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
            training_command += f'--adversarial-eps {adversarial_eps} --adversarial-eps-growth-start {adversarial_eps_growth_start} '
            training_command += f'--adversarial-eps-growth-epoch {adversarial_eps_growth_epoch} --checkpoint-every {checkpoint_every} '

        if test_name == 'relu':
            training_command += f'--l1-regularization {l1_regularization} --rs-regularization {rs_regularization} --rs-eps {rs_eps} '
            training_command += f'--rs-minibatch {rs_minibatch_size} --rs-start-epoch {rs_start_epoch} '

        print(f'Training | Running command\n"{training_command}"')
        os.system(training_command)

    if test_name == 'relu':
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
            relu_pruning_command += f'--deterministic --seed {seed}'

            print(f'ReLU Pruning | Running command\n"{relu_pruning_command}".')
            os.system(relu_pruning_command)

if __name__ == '__main__':
    main()