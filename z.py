import torch
import parsing
import attacks
model = parsing.parse_model('cifar10', 'c', 'trained-models/classifiers/cifar10-c-200.pth', True, False, False, load_weights=True)
dataset = parsing.parse_dataset('cifar10', 'std:test')
cpu_attack = attacks.get_carlini_linf_attack(model, 10, max_iterations=1000, tau_factor=0.75, const_factor=10, abort_early=False, return_best=True, cuda_optimized=False)
cuda_attack = attacks.get_carlini_linf_attack(model, 10, max_iterations=1000, tau_factor=0.75, const_factor=10, inner_check=1, abort_early=False, return_best=True, cuda_optimized=True)

images = torch.stack([dataset[i][0] for i in range(10)])
labels = torch.stack([torch.tensor(dataset[i][1]) for i in range(10)])
torch.manual_seed(0)
print('CPU')
#cpu_adversarials = cpu_attack.perturb(images, torch.argmax(model(images), dim=1))
cpu_adversarials = cpu_attack._run_attack(images, labels, 0.1, torch.tensor([0.02] * 10), torch.ones_like(images))
torch.manual_seed(0)
cpu_adversarials_1 = cpu_attack._run_attack(images, labels, 0.1, torch.tensor([0.002] * 10), torch.ones_like(cpu_adversarials))[-1]
torch.manual_seed(0)
cpu_adversarials_2 = cpu_attack._run_attack(images[-1].unsqueeze(0), labels[-1].unsqueeze(0), 0.1, torch.tensor([0.002]), torch.ones_like(cpu_adversarials)[-1].unsqueeze(0))
torch.manual_seed(0)
print('CUDA')
#cuda_adversarials = cuda_attack.perturb(images, torch.argmax(model(images), dim=1))
torch.manual_seed(0)
cuda_adversarials_1 = cuda_attack._run_attack(images, labels, 0.1, torch.tensor([0.002] * 10), torch.ones_like(cpu_adversarials))[-1]
torch.manual_seed(0)
cuda_adversarials_2 = cpu_attack._run_attack(images[-1].unsqueeze(0), labels[-1].unsqueeze(0), 0.1, torch.tensor([0.002]), torch.ones_like(cpu_adversarials)[-1].unsqueeze(0))
print(cpu_adversarials.shape)
torch.set_printoptions(precision=8)
print(torch.sum(torch.abs(cpu_adversarials_1 - cuda_adversarials_1)))
print(torch.sum(torch.abs(cpu_adversarials_2 - cuda_adversarials_2)))
print(torch.sum(torch.abs(cpu_adversarials_1 - cpu_adversarials_2)))
print(torch.sum(torch.abs(cuda_adversarials_1 - cuda_adversarials_2)))
