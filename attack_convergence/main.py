from email.policy import default
import sys
sys.path.append('.')

import json
import logging
from pathlib import Path

import click
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import attacks
from attack_convergence.core import ConvergenceAttack, ConvergenceWrapper, CustomIndexedDataset
import parsing
import utils

logger = logging.getLogger(__name__)

def run_one(domain, model, attack_name, override_number, track_every, attack_config_file, parameter_overrides, custom_stop=None):
    utils.set_seed(1) # 0 for tuning, 1 for actual tests

    # Prepare the attacks
    if custom_stop is None:
        stop = 101000
    else:
        stop = custom_stop

    wrapped_model = ConvergenceWrapper(model, track_every, stop=stop)
    attack_config = utils.read_attack_config_file(attack_config_file)

    if attack_name == 'brendel':
        # brendel needs custom support
        brendel_kwargs = attack_config.get_arguments(attack_name, domain, 'linf', 'standard')

        if parameter_overrides is not None:
            for key, value in parameter_overrides.items():
                brendel_kwargs[key] = value

        return_best = brendel_kwargs.pop('return_best')
        assert not return_best

        if 'init_attack' in brendel_kwargs:
            init_attack_name = brendel_kwargs.pop('init_attack')
            init_attack_parameter_overrides = brendel_kwargs.pop('init_attack_parameter_overrides', None)
            init_attack = parsing.parse_attack(init_attack_name, domain, float('inf'), 'standard', wrapped_model, attack_config, device='cuda',
                                       defended_model=None, seed=None, parameter_overrides=init_attack_parameter_overrides,
                                       suppress_blended_warning=True)
        else:
            init_attack = None

        attack = CustomBrendelBethgeAttack(wrapped_model, float('inf'), init_attack=init_attack, **brendel_kwargs)
    else:
        attack = parsing.parse_attack(attack_name, domain, float('inf'), 'standard', wrapped_model, attack_config, parameter_overrides=parameter_overrides, device='cuda')

    attack = ConvergenceAttack(wrapped_model, attack, float('inf'), False, suppress_warning=True)

    # Load the relevant indices
    with open(f'{domain}_indices_intersection.json') as f:
        relevant_indices = json.load(f)
    dataset = CustomIndexedDataset(parsing.parse_dataset(domain, 'std:test'), relevant_indices)
    dataloader = DataLoader(dataset, 250, num_workers=2, shuffle=False)

    results = { index: { 'stats' : [] } for index in relevant_indices}
    all_adversarials = {}

    for indices, (images, true_labels) in tqdm(dataloader, desc=f'{attack_name}-{override_number}'):
        indices = indices.cuda()
        images = images.cuda()
        true_labels = true_labels.cuda()
        images, true_labels, labels = utils.apply_misclassification_policy(model, images, true_labels, 'use_predicted')
        assert len(images) == len(indices)
        stats, _, batch_adversarials = attack(images, labels)
    
        for inner_index, actual_index in enumerate(indices):
            # inner_index: 0, 1, 2...
            # actual_index: 489, 491, 501...
            if isinstance(actual_index, torch.Tensor):
                actual_index = actual_index.item()

            for step, found, distances in stats:
                adversarial = batch_adversarials[inner_index] if found[inner_index].item() else None
                all_adversarials[actual_index] = adversarial

                results[actual_index]['stats'].append(
                    (step, distances[inner_index].item() if found[inner_index].item() else None)
                )

    return results, all_adversarials

# Variant of B&B that handles attack failures
# Note: B&B and Carlini were eventually dropped for this test due to the fact that they have
# different convergence rates depending on the batch size

class CustomBrendelBethgeAttack(attacks.BrendelBethgeAttack):
    def __init__(self, model, p, clip_min=0, clip_max=1, targeted=False, init_attack=None, overshoot: float = 1.1, steps: int = 1000, lr: float = 0.001, lr_decay: float = 0.5, lr_num_decay: int = 20, momentum: float = 0.8, tensorboard = False, binary_search_steps: int = 10, initialization_attempts=10, init_directions=1000, init_steps=1000):
        super().__init__(model, p, clip_min, clip_max, targeted, init_attack, overshoot, steps, lr, lr_decay, lr_num_decay, momentum, tensorboard, binary_search_steps, initialization_attempts, init_directions, init_steps)
        self.predict = model
    def perturb(self, x, y=None, starting_points=None):
        x, y = self._verify_and_process_inputs(x, y)

        for attempt in range(self.initialization_attempts):
            logger.debug(f'Initialization attempt {attempt + 1}.')

            if starting_points is None:
                starting_points = self.run_init_attack(x, y)
            else:
                assert starting_points.shape == x.shape

                # Foolbox's implementation of Brendel&Bethge requires all starting points to be successful
                # adversarials. Since that is not always the case, we use Brendel&Bethge's init_attack
                # to initialize the unsuccessful starting_points
                fallback = ~self.successful(starting_points, y)

                # We attack all samples to avoid issues during best sample tracking
                fallback_adversarials = self.run_init_attack(x, y)
                starting_points[fallback] = fallback_adversarials[fallback]

            successful_starting = self.successful(starting_points, y)

            num_failures = torch.count_nonzero(~successful_starting)

            if num_failures == 0:
                break

        if num_failures > 0:
            logger.warning(f'Failed to initialize {num_failures} starting points.')

        adversarials = torch.zeros_like(x)

        # For failed starting points, use the original images (which will be treated as failures)
        adversarials[~successful_starting] = x[~successful_starting]

        # For successful starting points, run the attack and store the results
        if torch.count_nonzero(successful_starting) > 0:
            if torch.count_nonzero(~successful_starting) > 0:
                self.predict.tracker.mask_default = successful_starting
            computed_adversarials = super().perturb(x[successful_starting], y=y[successful_starting], starting_points=starting_points[successful_starting])
            adversarials[successful_starting] = computed_adversarials
            self.predict.tracker.mask_default = None

        return adversarials

@click.command()
@click.argument('domain', type=click.Choice(['mnist', 'cifar10']))
@click.argument('architecture', type=click.Choice(['a', 'b', 'c']))
@click.argument('test_type', type=click.Choice(['standard', 'adversarial', 'relu']))
@click.argument('track_every', type=click.IntRange(1, None))
@click.option('--attack-config-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='balanced_attack_configuration.cfg', show_default=True, help='The path to the file containing the '
              'attack configuration.')
@click.option('--custom-stop', type=int, default=None)
def main(domain, architecture, test_type, track_every, attack_config_file, custom_stop):
    utils.enable_determinism()

    # Load the model
    if test_type == 'relu':
        state_dict_path = f'trained-models/classifiers/relu/relu-pruned/{domain}-{architecture}.pth'
    else:
        state_dict_path = f'trained-models/classifiers/{test_type}/{domain}-{architecture}.pth'

    model = parsing.parse_model(domain, architecture, state_dict_path, False, test_type == 'relu', False, True)
    model.eval()
    model.cuda()

    with open(f'attack_convergence/best_overrides_{domain}.json') as f:
        overrides = json.load(f)

    # Note the lack of 'brendel' and 'carlini' (which have different convergence rates depending on the batch size)
    for attack_name in ['bim', 'deepfool', 'fast_gradient', 'pgd', 'uniform']:
        for override_number in [100, 1000, 10000]:
            try:
                if custom_stop is None:
                    save_path = f'attack_convergence/attack_stats/{domain}/{architecture}/{test_type}/{attack_name}/{override_number}.json'
                    adversarial_path = f'attack_convergence/attack_adversarials/{domain}/{architecture}/{test_type}/{attack_name}/{override_number}.pt'
                else:
                    save_path = f'attack_convergence/attack_stats/{custom_stop}/{domain}/{architecture}/{test_type}/{attack_name}/{override_number}.json'
                    adversarial_path = f'attack_convergence/attack_adversarials/{custom_stop}/{domain}/{architecture}/{test_type}/{attack_name}/{override_number}.pt'

                save_path = Path(save_path)
                adversarial_path = Path(adversarial_path)

                if not save_path.exists() or not adversarial_path.exists():
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    adversarial_path.parent.mkdir(parents=True, exist_ok=True)

                    override = overrides[attack_name][str(override_number)]
                    results, adversarials = run_one(domain, model, attack_name, override_number, track_every, attack_config_file, override, custom_stop=custom_stop)

                    with open(str(save_path), 'w') as f:
                        json.dump(results, f)
                    torch.save(adversarials, str(adversarial_path))
            except:
                import traceback
                traceback.print_exc()

if __name__ == '__main__':
    main()