import torch
import parsing
import attacks
model = parsing.parse_model('cifar10', 'c', 'trained-models/classifiers/cifar10-c-200.pth', True, False, False, load_weights=True)
dataset = parsing.parse_dataset('cifar10', 'std:test')
cpu_attack = attacks.get_carlini_linf_attack(model, 10, max_iterations=100, return_best=False, abort_early=False, cuda_optimized=False)
cuda_attack = attacks.get_carlini_linf_attack(model, 10, max_iterations=100, return_best=False, abort_early=False, inner_check=1, cuda_optimized=True)

torch.set_printoptions(precision=8)

images = torch.stack([dataset[i][0] for i in range(10)])
"""
print(model(images)[-1])
print(model(images[-1].unsqueeze(0)))

raise RuntimeError()
print(torch.sum(model(images[-1].unsqueeze(0))))
print(torch.sum(model(torch.zeros_like(images)[-1].unsqueeze(0))))

raise RuntimeError()"""

torch.manual_seed(0)
print('CPU')
cpu_adversarials = cpu_attack.perturb(images, torch.argmax(model(images), dim=1))
torch.manual_seed(0)
print('CUDA')
cuda_adversarials = cuda_attack.perturb(images, torch.argmax(model(images), dim=1))
print(cpu_adversarials.shape)
print(torch.max(torch.abs(cpu_adversarials - cuda_adversarials)))
print([torch.max(torch.abs(cpu_adv - cuda_adv)) for cpu_adv, cuda_adv in zip(cpu_adversarials, cuda_adversarials)])
print([torch.mean(torch.abs(cpu_adv - cuda_adv)) for cpu_adv, cuda_adv in zip(cpu_adversarials, cuda_adversarials)])
print([torch.count_nonzero(torch.abs(cpu_adv - cuda_adv)) for cpu_adv, cuda_adv in zip(cpu_adversarials, cuda_adversarials)])
print([torch.sum(adv) for adv in cpu_adversarials])
print([torch.sum(adv) for adv in cuda_adversarials])