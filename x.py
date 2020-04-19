# Torch ha una sezione che converte in ONNX
# https://github.com/sisl/NNet/blob/master/converters/onnx2nnet.py converte da ONNX a NNet
# https://github.com/guykatzz/ReluplexCav2017 esegue ReLUPlex
# https://github.com/huanzhang12/RecurJac-and-CROWN lavora in NumPy
# https://github.com/huanzhang12/CROWN-IBP lavora direttamente su PyTorch
# https://github.com/NeuralNetworkVerification/Marabou Questo sembra mantenuto attivamente (e supporta Python)
# Credo però che solo ReLUPlex dia veri bound esatti (i bound degli altri sembrano più loose)
# Controllare se esistono altre implementazioni!

# TODO: Marabou non supporta GEMM con transB=1
# TODO: Marabou ha un bug per Transpose

# Aggiungere domain mini-mnist?

# Marabou legge direttamente ONNX o NNET

import copy
import os
import sys
import time

marabou_folder = './Marabou'
sys.path.append(marabou_folder)

#import Marabou.maraboupy as marabou
from maraboupy import Marabou
from maraboupy import MarabouCore
from Marabou.maraboupy.MarabouUtils import addInequality

import numpy as np

import torch
import torch.nn as nn

import onnx
import onnxruntime
import onnx2nnet

import matplotlib.pyplot as plt


n_input = 28 * 28
n_hidden = 24
n_output = 10

simple_network = nn.Sequential(
    nn.Linear(n_input, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, n_output),
)

"""small_n_hidden = 24

simple_network = nn.Sequential(
    nn.Linear(n_input, small_n_hidden),
    nn.ReLU(),
    nn.Linear(small_n_hidden, small_n_hidden),
    nn.Linear(small_n_hidden, n_output)
)"""

class ReshapedNetwork(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        x = x.reshape(1, 28 * 28)
        return self.network(x)

batch_size = 1 # Perché dà problemi con batch size >1?

simple_network = ReshapedNetwork(simple_network)

x = torch.rand((batch_size, 28, 28), requires_grad=True)
torch_out = simple_network(x)


"""

#class AdversarialAttackNetwork()


#simple_network._initialize_weights()
simple_network.eval()

onnx_path = 'exported_simple_network.onnx'
nnet_path = 'nnet_simple_network.nnet'


# Export the model
torch.onnx.export(simple_network,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})


onnx_model = onnx.load(onnx_path)

new_nodes = []

print([x.name for x in onnx_model.graph.initializer])
for i, node in enumerate(onnx_model.graph.node):
    
    #node.name = str(i)
    if node.op_type == 'Gemm':
        transB = next(x for x in node.attribute if x.name=='transB')
        print('========')

        if transB.i == 1:
            b_tensor = next(x for x in onnx_model.graph.initializer if x.name == node.input[1])
            b = node.input[1]
            transpose_node = onnx.helper.make_node('Transpose', [b], [b_tensor.name + '_transposed'], name=b + '_transposed', perm=(1, 0))
            print('{} -> {} -> {}'.format(b, transpose_node.name, node.name))
            node.input[1] = transpose_node.output[0]
            transB.i = 0
            new_nodes.append(transpose_node)
        
    new_nodes.append(node)

print([x.name for x in onnx_model.graph.node])

new_graph = onnx.helper.make_graph(new_nodes, onnx_model.graph.name,
onnx_model.graph.input, onnx_model.graph.output, list(onnx_model.graph.initializer),
onnx_model.graph.doc_string, onnx_model.graph.value_info)

onnx_model = onnx.helper.make_model(new_graph, model_version=onnx_model.model_version, domain=onnx_model.domain, ir_version=onnx_model.ir_version)
print([x.name for x in onnx_model.graph.node])
#onnx_model = onnx.helper.make_model(new_graph, producer_name=onnx_model.)

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, onnx_path)



ort_session = onnxruntime.InferenceSession(onnx_path)



# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
"""

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def misclassified_constraint(network, output_vars, correct_label):
    correct_variable = output_vars[correct_label]

    other_labels = [label for label in range(len(output_vars)) if label != correct_label]
    other_variables = [output_vars[label] for label in other_labels]

    max_other_variables = network.getNewVariable()

    network.addMaxConstraint(set(other_variables), max_other_variables)

    # correct - max(others) <= 0
    addInequality(network, [correct_variable, max_other_variables], [1, -1], 0)

def distance_constraint(network, input_vars, original, delta, clip_min=0, clip_max=1):
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            index = y * original.shape[1] + x

            min_ = max(original[y, x].item() - delta, clip_min)
            max_ = min(original[y, x].item() + delta, clip_max)

            assert max_ > min_

            network.setLowerBound(input_vars[y][x], min_)
            network.setUpperBound(input_vars[y][x], max_)



#marabou_network = Marabou.read_onnx(onnx_path)
from certified_attack import torch_to_marabou

marabou_network = torch_to_marabou(simple_network, x)

options = MarabouCore.Options()
options._verbosity = 0

inputVars = marabou_network.inputVars[0][0][0]
outputVars = marabou_network.outputVars

print(inputVars)
#print(outputVars)

marabouEval = marabou_network.evaluateWithMarabou([to_numpy(x)], options=options)
print(marabouEval)
print(marabouEval - to_numpy(torch_out))
print()
onnxEval = marabou_network.evaluateWithoutMarabou([to_numpy(x)])
print(onnxEval)
print(onnxEval - to_numpy(torch_out))

"""
# Come gestire il preprocessing?
for i in range(len(inputVars)):
        marabou_network.setLowerBound(inputVars[i], 0)
        marabou_network.setUpperBound(inputVars[i], 1)

for i in range(len(outputVars)):
    marabou_network.setLowerBound(outputVars[i], 0)
    marabou_network.setUpperBound(outputVars[i], 5)"""

original = np.random.rand(28, 28)

original_output = marabou_network.evaluateWithMarabou([original], options=options)
original_label = np.argmax(original_output).item()

precision = 0.001
min_delta = 0
max_delta = 1

import certified_attack

certified_attack.marabou_attack(marabou_network, original, original_label, 0.5, inputVars, outputVars)
"""
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

best_adversarial = np.zeros_like(original)

while abs(min_delta - max_delta) > precision:
    copied_network = copy.deepcopy(marabou_network)
    delta = (min_delta + max_delta) / 2

    input_vars = copied_network.inputVars[0][0]
    output_vars = copied_network.outputVars

    distance_constraint(copied_network, input_vars, original, delta)

    misclassified_constraint(copied_network, output_vars, original_label)

    print('Testing with delta={}'.format(delta))
    start_time = time.time()
    with HiddenPrints():
        vals, stats = copied_network.solve(options=options)
    
    print('Elapsed time: {}s'.format(time.time() - start_time))

    if stats.hasTimedOut():
        print('Timeout!')
        break

    if len(vals) > 0:
        # Success: Found an adversarial, save it and decrease the range
        for y in range(best_adversarial.shape[0]):
            for x in range(best_adversarial.shape[1]):
                index = y * best_adversarial.shape[1] + x
                best_adversarial[y, x] = vals[input_vars[y][x]]

        max_delta = delta
    else:
        # Failure: increase the range
        min_delta = delta

image = np.zeros((28, 28))"""

#print('{} X'.format(i))
#print(inputVars)

# print('Vals: {}'.format(vals))
#print('Stats: {}'.format(stats))

# Marabou supporta max(x1, ..., xn)

# Possibile metodo per supportare "l'output è lo stesso":
# max([target - y1, ..., target - yn]) < 0
# Ricordati di partire da un adversarial!

# Per distance constraint: x0 - k < x < x0 + k