import onnx
import torch
import torch.nn as nn

n_input = 28 * 28
n_hidden = 24
n_output = 10

batch_size = 10

simple_network = nn.Sequential(
    nn.Flatten(),
    nn.Linear(n_input, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_output),
)

class ReshapedNetwork(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        x = x.reshape(batch_size, -1)
        return self.network(x)

#simple_network = ReshapedNetwork(simple_network)

simple_network.eval()

onnx_path = 'exported_simple_network.onnx'

dummy_input = torch.rand((batch_size, 28, 28), requires_grad=True)

# Export the model
torch.onnx.export(simple_network,
                  dummy_input,
                  onnx_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})

onnx_model = onnx.load(onnx_path)

for node in onnx_model.graph.node:
    print(node.op_type)

import sys
marabou_folder = './Marabou'
sys.path.append(marabou_folder)

julia_folder = 'C:\\Users\\Samuele\\AppData\\Local\\Programs\\Julia\\Julia-1.4.1\\bin'
sys.path.append(julia_folder) # TODO: Non sembra funzionare?

#print(sys.path)

# TODO: Probabilmente dovrei lasciare l'installazione a livello individuale

import julia
julia.install() # Skipped if PyJulia is already installed
from julia import Base

from julia import Pkg

#Pkg.add('Gurobi') # TODO: Support for arbitrary solver
#Pkg.add('MIPVerify')
#Pkg.add('JuMP')

# TODO: Serve anche MAT?

from julia import MIPVerify
from julia import Main
from julia import Gurobi
from julia import JuMP

#Base.include("z.jl")
#Main.include('z.jl')

# TODO: Passare tolerance?


def module_to_mip(module):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    if isinstance(module, nn.Linear):
        converted = MIPVerify.Linear(
            to_numpy(module.weight).transpose(),
            to_numpy(module.bias)
        )
    elif isinstance(module, nn.Flatten):
        assert module.end_dim == -1
        converted = MIPVerify.Flatten(2) # TODO: Con che parametri?
    elif isinstance(module, nn.ReLU):
        converted = MIPVerify.ReLU()
    else:
        raise NotImplementedError('Unsupported module.') # TODO: Supporto altri moduli?

    return converted

def sequential_to_mip(sequential):
    converted_layers = []
    for layer in sequential:
        converted_layers.append(module_to_mip(layer))

    return MIPVerify.Sequential(converted_layers, 'Converted network')


#converted_linear = MIPVerify.Linear(test_linear.weight.detach().cpu().numpy().transpose(), test_linear.bias.detach().cpu().numpy())

converted_network = sequential_to_mip(simple_network)

import numpy as np
original = np.random.rand(28, 28)

original_output = simple_network(torch.from_numpy(original).float().unsqueeze(0))
original_label = np.argmax(original_output.detach().cpu().numpy()).item()

print('Original label: {}'.format(original_label))

import certified_attack

attack = certified_attack.MIPAttack(simple_network, float('inf'), 10, False)
adversarials = attack(torch.from_numpy(original).unsqueeze(0).float())
adversarial_output = simple_network(adversarials)
adversarial_label = np.argmax(adversarial_output.detach().cpu().numpy()).item()
print('Adversarial label: {}'.format(adversarial_label))

"""
# Note: Julia is 1-indexed and column-major
target_label = original_label#(original_label + 1) % 10

# TODO: Gli puoi passare il Linf bound (magari ottenuto da un initialisation attack)

# TODO: Di default lavora in L1
# TODO: How to use untargeted attacks? With invert_target_selection?
# TODO: Testare entrambi in forma untargeted
# Nota: invert_target_selection sembra non funzionare
untargeted_label = [x for x in range(10) if x != original_label]

#Main.untargeted_label = [x + 1 for x in untargeted_label]
#Main.original = original
#Main.converted_network = converted_network
#Main.solver = Gurobi.GurobiSolver(OutputFlag=0)
#Main.norm_order = float('inf')

#from julia import CounterAttack

adversarial_result = MIPVerify.find_adversarial_example(converted_network, original, [x + 1 for x in untargeted_label], Gurobi.GurobiSolver(OutputFlag=0), norm_order=float('inf'),
    )
#adversarial_result = MIPVerify.find_adversarial_example(converted_network, original, original_label, Gurobi.GurobiSolver(OutputFlag=0), invert_target_selection=True, norm_order=float('inf'),
#    )

#Main.eval('result = CounterAttack.find_adversarial_example_with_bounds(converted_network, original, untargeted_label, solver, norm_order=norm_order)')
#adversarial_result = Main.result

#adversarial = np.array(adversarial['PerturbedInput'])

adversarial = np.array(JuMP.getvalue(adversarial_result['PerturbedInput']))

adversarial_output = simple_network(torch.from_numpy(adversarial).float().unsqueeze(0))
adversarial_label = np.argmax(adversarial_output.detach().cpu().numpy()).item()

print('Adversarial label: {}'.format(adversarial_label))

# Nota: Supporta "normalise"

# TODO: Check if the adversarial is within bounds

from maraboupy import Marabou
marabou_network = Marabou.read_onnx(onnx_path)

a = marabou_network.inputVars
print(a[0].shape)
print(marabou_network.outputVars.shape)
# Prima dimensione: permette di gestire input separati (ed è una lista)
# Seconda dimensione: legata alla batch size? Forse
# Terza e quarta dimensione: dimensione dell'immagine

# Per l'output non sembra esserci una dimensione dedicata ai possibili output


import certified_attack


attack = certified_attack.CertifiedLinfAttack(simple_network)

adversarial = attack(dummy_input)"""