from pathlib import Path
import sys
from time import time

sys.path.append('.')

import json
import logging

import advertorch.attacks
import click
import torch
import torch.nn as nn

import attacks
from fooling.estimators import NESWrapper
import parsing
import utils

ENABLE_SUCCESS_ONLY = False


class AttackSuccessException(Exception):
    pass

logger = logging.getLogger(__name__)

def batch_distance(first, second): # Linf distances
    distances = torch.abs(first - second)
    distances = torch.flatten(distances, start_dim=1)
    distances, _ = torch.max(distances, dim=1, keepdim=True)

    return distances

def get_results(genuine, adversarial, output, predicted_label, rejection_distance, elapsed_time):
    success = check_success(output, predicted_label, rejection_distance).item()

    if ENABLE_SUCCESS_ONLY and success:
        raise AttackSuccessException()

    # print(adversarial.shape)
    # print(genuine.shape)
    adversarial_distance = torch.max(torch.abs(adversarial - genuine))
    estimated_distance = torch.squeeze(output[:, -1]).item()
    results = {
        'success' : success,
        'elapsed' : elapsed_time,
        'adversarial_distance' : adversarial_distance.item() if success else None,
        'estimated_distance' : estimated_distance if success else None,
    }
    adversarial_results = adversarial if success else None

    return results, adversarial_results

def split_outputs(outputs):
    assert len(outputs.shape) == 2
    base_classes = outputs.shape[1] - 1
    base_outputs = outputs[:, :base_classes]
    distances = outputs[:, base_classes]
    return base_outputs, distances

def check_success(outputs, labels, epsilon):
    base_outputs, distances = split_outputs(outputs)
    # Fools the model and has an over-estimated decision boundary distance
    success = ~torch.eq(torch.argmax(base_outputs, dim=1), labels) & (distances > epsilon)

    if ENABLE_SUCCESS_ONLY and torch.any(success):
        print('Success!')
        raise AttackSuccessException()

    return success

class CCALoss(nn.Module):
    def __init__(self, base_loss, coefficient):
        super().__init__()
        self.base_loss = base_loss
        self.coefficient = coefficient
    
    def forward(self, outputs, y):
        base_outputs, distances = split_outputs(outputs)

        return self.base_loss(base_outputs, y) + self.coefficient * distances # Both with the same sign

class CAModel(nn.Module):
    def __init__(self, base_model, input_size, attack, nb_samples, fd_eta, estimate_grad=True):
        super().__init__()
        self.base_model = base_model
        self.input_size = input_size
        self.attack = attack

        def ca_distances(x):
            x = x.reshape([x.shape[0]] + self.input_size)
            with torch.enable_grad():
                ca_adversarials = self.attack(x)
            return batch_distance(x, ca_adversarials)

        if estimate_grad:
            self.distance_estimator = NESWrapper(ca_distances, nb_samples, fd_eta)
        else:
            self.distance_estimator = ca_distances

    def get_distances(self, x):
        flattened_x = torch.flatten(x, start_dim=1)
        distances = self.distance_estimator(flattened_x)
        return distances.reshape(x.shape[0])

    def forward(self, x):
        base_outputs = self.base_model(x)
        distances = self.get_distances(x)

        merged_outputs = torch.cat([base_outputs, torch.unsqueeze(distances, 1)], dim=1)
        assert tuple(merged_outputs.shape) == (base_outputs.shape[0], base_outputs.shape[1] + 1)
        return merged_outputs

class CABestSampleWrapper(attacks.BestSampleWrapper):
    def __init__(self, model, max_ca_distance):
        super().__init__(model)
        self.max_ca_distance = max_ca_distance

    def successful(self, outputs, relevant_labels):
        base_outputs, distances = split_outputs(outputs)
        # print(distances.item(), torch.eq(torch.argmax(base_outputs), relevant_labels).item())
        # Fools the model and has an over-estimated decision boundary distance
        if (super().successful(base_outputs, relevant_labels) & (distances > self.max_ca_distance)).item():
            if ENABLE_SUCCESS_ONLY:
                print('Success!')
                raise AttackSuccessException()
        return super().successful(base_outputs, relevant_labels) & (distances > self.max_ca_distance)

class CAUniformNoiseAttack(attacks.UniformNoiseAttack):
    def __init__(self, predict, p, targeted, max_ca_distance, eps=0.3, count=100, clip_min=0, clip_max=1):
        super().__init__(predict, p, targeted, eps, count, clip_min, clip_max)
        self.max_ca_distance = max_ca_distance
    
    def successful(self, adversarials, y):
        outputs = self.predict(adversarials)

        base_outputs, distances = split_outputs(outputs)
        predicted_labels = torch.argmax(base_outputs, dim=1)
        if self.targeted:
            base_successful = torch.eq(predicted_labels, y)
        else:
            base_successful = ~torch.eq(predicted_labels, y)
        
        final_successful = base_successful & (distances > self.max_ca_distance)

        if ENABLE_SUCCESS_ONLY and torch.any(final_successful):
            print('Success!')
            raise AttackSuccessException()

        return final_successful


"""class CAEpsilonBinarySearchAttack(attacks.EpsilonBinarySearchAttack):
    def __init__(self, inner_attack, p, max_ca_distance, targeted=False, min_eps=0, max_eps=1, eps_initial_search_steps=9, eps_initial_search_factor=0.5, eps_binary_search_steps=9):
        super().__init__(inner_attack, p, targeted, min_eps, max_eps, eps_initial_search_steps, eps_initial_search_factor, eps_binary_search_steps)
        self.max_ca_distance = max_ca_distance
    
    def successful(self, adversarials, y):
        outputs = self.predict(adversarials)

        base_outputs, distances = split_outputs(outputs)

        adversarial_labels = torch.argmax(base_outputs, axis=1)
        assert adversarial_labels.shape == y.shape

        if self.targeted:
            successful = torch.eq(adversarial_labels, y)
        else:
            successful = ~torch.eq(adversarial_labels, y)
        
        return successful & distances <= self.max_ca_distance # Incorretto"""

# Non serve, Deepfool è deterministico
"""class ConsistentRandomStateWrapper(nn.Module):
    def __init__(self, predict, seed):
        super().__init__()
        self.predict = predict
        self.seed = seed
    
    def forward(self, x):
        random_state = utils.get_rng_state()
        utils.set_seed(self.seed)
        adversarials = self.predict(x)
        utils.set_rng_state(random_state)
        return adversarials"""

# Two variants: epsilon1 = epsilon2 and epsilon1 > epsilon2

def run_one(domain, model, dataset, index, input_size, rejection_distance, eps_correction):
    utils.set_seed(1)

    if eps_correction is None:
        search_distance = rejection_distance
    else:
        search_distance = rejection_distance * eps_correction
    attack_config = utils.read_attack_config_file('balanced_attack_configuration.cfg')

    with open(f'attack_convergence/best_overrides_{domain}.json') as f:
        overrides = json.load(f)
    deepfool_overrides = overrides['deepfool']['1000']

    counter_attack = parsing.parse_attack('deepfool', domain, float('inf'), 'standard', model, attack_config, 'cuda', parameter_overrides=deepfool_overrides)

    num_estimates = 200
    estimate_eps = 1e-4
    print('Target distance:', rejection_distance)

    genuine = torch.unsqueeze(dataset[index][0], dim=0).to('cuda')
    # Note that this is done before adding the extra output (the decision boundary distance)
    predicted_label = torch.argmax(model(genuine), dim=1)

    wrapped_model = CAModel(model, input_size, counter_attack, num_estimates, estimate_eps)
    # wrapped_model = ConsistentRandomStateWrapper(wrapped_model, seed=inner_seed)
    wrapped_model_best = CABestSampleWrapper(wrapped_model, rejection_distance)

    wrapped_model_no_grad = CAModel(model, input_size, counter_attack, None, None, estimate_grad=False)
    # wrapped_model_no_grad = ConsistentRandomStateWrapper(wrapped_model_no_grad, seed=inner_seed)
    wrapped_model_no_grad_best = CABestSampleWrapper(wrapped_model_no_grad, rejection_distance)

    results = {}
    adversarial_results = {}

    success_flag = False

    for coefficient in [1e4, 1e2, 1e0, 1e-2, 1e-4]:
        if not success_flag:
            print('Current coefficient:', coefficient)
            try:
                loss_function = CCALoss(nn.CrossEntropyLoss(reduction='sum'), coefficient)
                # 40 is PGD's default iteration number
                outer_attack = advertorch.attacks.LinfPGDAttack(wrapped_model_best, loss_function, eps=search_distance, nb_iter=400, eps_iter=0.001)
                outer_attack = attacks.BestSampleAttack(wrapped_model_best, outer_attack, float('inf'), False)

                start_time = time()
                adversarial = torch.squeeze(outer_attack(genuine, y=predicted_label))
                elapsed_time = time() - start_time
                # print('Elapsed:', elapsed_time)
                output = wrapped_model(torch.unsqueeze(adversarial, 0))
                # print('Output:', output)
                results[coefficient], adversarial_results[coefficient] = get_results(genuine, adversarial, output, predicted_label, rejection_distance, elapsed_time)
            except AttackSuccessException:
                success_flag = True

    if not success_flag:
        try:
            outer_uniform_attack = CAUniformNoiseAttack(wrapped_model_no_grad_best, float('inf'), False, max_ca_distance=rejection_distance, eps=search_distance, count=1000)
            outer_uniform_attack = attacks.BestSampleAttack(wrapped_model_no_grad_best, outer_uniform_attack, float('inf'), False)

            start_time = time()
            uniform_adversarial = torch.squeeze(outer_uniform_attack(genuine))
            elapsed_time = time() - start_time
            uniform_output = wrapped_model_no_grad(torch.unsqueeze(uniform_adversarial, 0))
            results['uniform'], adversarial_results['uniform'] = get_results(genuine, uniform_adversarial, uniform_output, predicted_label, rejection_distance, elapsed_time)
            uniform_success = check_success(uniform_output, predicted_label, rejection_distance).item()
        #print('Uniform success:', success)
        except AttackSuccessException:
            success_flag = True

    if ENABLE_SUCCESS_ONLY:
        return success_flag, None
    else:
        print(results)
        return results, adversarial_results


@click.command()
@click.argument('domain')
@click.argument('architecture')
@click.argument('test_type')
@click.argument('batch_size', type=int)
@click.argument('batch_index', type=int)
@click.option('--eps-correction', type=float, default=None)
@click.option('--success-only', is_flag=True)
def main(domain, architecture, test_type, batch_size, batch_index, eps_correction, success_only):
    utils.enable_determinism()
    utils.set_seed(1) # 0 is for tuning, 1 is for actual tests

    global ENABLE_SUCCESS_ONLY
    ENABLE_SUCCESS_ONLY = success_only

    if ENABLE_SUCCESS_ONLY:
        print('Running in ENABLE_SUCCESS_ONLY mode. No additional data will be collected.')
    if eps_correction:
        print('Using eps correction', str(eps_correction))

    if domain == 'mnist':
        input_size = [1, 28, 28]
        epsilons = [0.025, 0.05, 0.1]
    else:
        input_size = [3, 32, 32]
        epsilons = [2/255, 4/255, 8/255]

    if test_type == 'relu':
        model_path = f'trained-models/classifiers/relu/relu-pruned/{domain}-{architecture}.pth'
    else:
        model_path = f'trained-models/classifiers/{test_type}/{domain}-{architecture}.pth'
    model = parsing.parse_model(domain, architecture, model_path, False, False, False, True)
    model.eval()
    model.cuda()

    dataset = parsing.parse_dataset(domain, 'std:test')

    with open(f'fooling/sorted_indices_{domain}.json') as f:
        index_list = json.load(f)

    for i, index in enumerate(index_list):
        if i % batch_size != batch_index:
            print('Skipping', index, 'due to wrong index')
            continue

        if eps_correction is None:
            results_path = Path(f'fooling/results/{domain}/{architecture}/{test_type}/{index}.json')
            adversarials_path = Path(f'fooling/adversarials/{domain}/{architecture}/{test_type}/{index}.pt')
        else:
            results_path = Path(f'fooling/results/{eps_correction}/{domain}/{architecture}/{test_type}/{index}.json')
            adversarials_path = Path(f'fooling/adversarials/{eps_correction}/{domain}/{architecture}/{test_type}/{index}.pt')

        if results_path.exists() and (ENABLE_SUCCESS_ONLY or adversarials_path.exists()):
            print(f'Skipping {index} due to the fact that it already exists (success_only = {ENABLE_SUCCESS_ONLY})')
        else:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            if not ENABLE_SUCCESS_ONLY:
                adversarials_path.parent.mkdir(parents=True, exist_ok=True)

            print('Running with index', index)
            global_results = {}
            global_adversarials = {}
            start_time = time()

            for epsilon in epsilons:
                global_results[epsilon], global_adversarials[epsilon] = run_one(domain, model, dataset, index, input_size, epsilon, eps_correction)
            
            total_elapsed_time = time() - start_time

            global_results['total_elapsed_time'] = total_elapsed_time

            with open(str(results_path), 'w') as f:
                json.dump(global_results, f)
            
            if not ENABLE_SUCCESS_ONLY:
                torch.save(global_adversarials, adversarials_path)

if __name__ == '__main__':
    main()