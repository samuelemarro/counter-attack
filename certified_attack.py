import copy
import logging
from pathlib import Path
import time

import advertorch
import numpy as np
import onnx
import onnxruntime
import torch

from maraboupy import Marabou
from maraboupy import MarabouCore
from Marabou.maraboupy.MarabouUtils import addInequality

import utils

logger = logging.getLogger(__name__)

def fix_onnx(onnx_model):
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

    return onnx.helper.make_model(new_graph, model_version=onnx_model.model_version, domain=onnx_model.domain, ir_version=onnx_model.ir_version)


# TODO: Gestire la forma di input e output







# TODO: Scelta targeted/untargeted?

class CertifiedLinfAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, initialisation_attack=None, min_delta=0, max_delta=1, precision=1e-3, clip_min=0, clip_max=1, fail_on_timeout=True, temp_path='./tmp'):
        super().__init__(predict, None, clip_min, clip_max)
        self.initialisation_attack = initialisation_attack
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.precision = precision
        self.fail_on_timeout = fail_on_timeout
        self.temp_path = temp_path

        self.marabou_model = None

        self.targeted = False # Always false

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        # Initialization
        # TODO: Ridondante?
        if y is None:
            y = self._get_predicted_label(x)

        adversarials = []

        for image, label in zip(x, y):
            if self.marabou_model is None:
                self.marabou_model = self.torch_to_marabou(self.predict, image.unsqueeze(0))

            image = image.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            
            adversarial = self.marabou_attack(image, label)
            adversarials.append(torch.from_numpy(adversarial).to(x.device))

        # TODO: Finire
        return utils.maybe_stack(adversarials, x.shape[1:], device=x.device)

    def marabou_attack(self, genuine, label):
        best_adversarial = genuine.copy()

        options = MarabouCore.Options()
        options._verbosity = 0

        min_delta = self.min_delta
        max_delta = self.max_delta

        while abs(min_delta - max_delta) > self.precision:
            copied_network = copy.deepcopy(self.marabou_model)

            # Only one possible input (the image)
            assert len(copied_network.inputVars) == 1

            input_vars = np.squeeze(copied_network.inputVars[0])
            output_vars = np.squeeze(copied_network.outputVars)

            delta = (min_delta + max_delta) / 2

            self.distance_constraint(copied_network, input_vars, genuine, delta)

            self.misclassified_constraint(copied_network, output_vars, label)

            print('Testing with delta={}'.format(delta))
            start_time = time.time()

            with utils.HiddenPrint():
                vals, stats = copied_network.solve(options=options)
            
            logger.debug('Elapsed time: {}s'.format(time.time() - start_time))

            if stats.hasTimedOut():
                logger.info('Timeout!')
                if self.fail_on_timeout:
                    return genuine
                else:
                    break

            if len(vals) > 0:
                # Success: Found an adversarial, save it and decrease the range
                for y in range(best_adversarial.shape[0]):
                    for x in range(best_adversarial.shape[1]):
                        # TODO: Supporto matrici 3D
                        index = y * best_adversarial.shape[1] + x
                        best_adversarial[y, x] = vals[input_vars[y][x]]

                max_delta = delta
            else:
                # Failure: increase the range
                min_delta = delta

        return best_adversarial

    def misclassified_constraint(self, network, output_vars, correct_label):
        correct_variable = output_vars[correct_label]

        other_labels = [label for label in range(len(output_vars)) if label != correct_label]
        other_variables = [output_vars[label] for label in other_labels]

        max_other_variables = network.getNewVariable()

        network.addMaxConstraint(set(other_variables), max_other_variables)

        # correct - max(others) <= 0
        addInequality(network, [correct_variable, max_other_variables], [1, -1], 0)

    def distance_constraint(self, network, input_vars, original, delta):
        for y in range(original.shape[0]):
            for x in range(original.shape[1]):
                index = y * original.shape[1] + x

                min_ = max(original[y, x].item() - delta, self.clip_min)
                max_ = min(original[y, x].item() + delta, self.clip_max)

                assert max_ > min_

                network.setLowerBound(input_vars[y][x], min_)
                network.setUpperBound(input_vars[y][x], max_)


    def torch_to_marabou(self, model, reference_input, temp_path='./tmp'):
        model.eval()

        temp_path = Path(temp_path)
        temp_path.mkdir(parents=True, exist_ok=True)

        onnx_path = str(temp_path / 'model.onnx')

        x = reference_input.detach().clone()
        numpy_x = reference_input.clone().detach().cpu().numpy()
        x.requires_grad = True

        reference_output = model(x)[0].detach().cpu().numpy()

        # Export the model
        torch.onnx.export(model,               # model being run
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

        #onnx_model = fix_onnx(onnx_model)

        onnx.checker.check_model(onnx_model)
        #onnx.save(onnx_model, onnx_path)

        ort_session = onnxruntime.InferenceSession(onnx_path)

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: numpy_x}
        ort_outputs = ort_session.run(None, ort_inputs)

        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(reference_output, ort_outputs[0][0], rtol=1e-03, atol=1e-05)

        marabou_network = Marabou.read_onnx(onnx_path)

        options = MarabouCore.Options()
        options._verbosity = 0

        marabou_output = marabou_network.evaluateWithMarabou([numpy_x], options=options)
        
        np.testing.assert_allclose(reference_output, marabou_output, rtol=1e-03, atol=1e-05)
        
        marabou_onnx_output = marabou_network.evaluateWithoutMarabou([numpy_x])
        np.testing.assert_allclose(reference_output, marabou_onnx_output[0], rtol=1e-03, atol=1e-05)

        return marabou_network


import torch.nn as nn
from julia import MIPVerify
from julia import JuMP

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

#TODO: Come gestire il fatto che si ottengono output esatti, che possono quindi coincidere?
# Opzione 1: Mettere tolerance
# Opzione 2: Se la label è la stessa, verificare se la c'è un output molto vicino
# Opzione 3: Se la label è la stessa, riprovare con tolerance > 0

class MIPAttack(advertorch.attacks.Attack, advertorch.attacks.LabelMixin):
    def __init__(self, predict, p, num_labels, targeted, tolerance=1e-6, clip_min=0, clip_max=1, solver='gurobi'):
        super().__init__(predict, None, clip_min, clip_max)
        if p in [1, 2, float('inf')]:
            self.p = p
        else:
            raise NotImplementedError('MIPAttack only supports p=1, 2 or inf.')

        self.num_labels = num_labels
        self.mip_model = sequential_to_mip(predict)
        self.targeted = targeted
        self.tolerance = tolerance

        if solver == 'gurobi':
            from julia import Gurobi
            self.solver = Gurobi.GurobiSolver(OutputFlag=0)
        else:
            raise NotImplementedError('Unsupported solver "{}".'.format(solver))

    def mip_attack(self, image, label):
        # Julia is 1-indexed
        if self.targeted:
            mip_target_label = label + 1
        else:
            target_labels = [x + 1 for x in range(self.num_labels) if x != label]

        adversarial_result = MIPVerify.find_adversarial_example(self.mip_model,
                                    image, target_labels, self.solver, norm_order=self.p,
                                    tolerance=self.tolerance)

        adversarial = np.array(JuMP.getvalue(adversarial_result['PerturbedInput']))

        return adversarial

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)

        adversarials = []

        for image, label in zip(x, y):
            print(image.dtype)
            image = image.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

            print(image.dtype)
            print(label.dtype)
            
            adversarial = self.mip_attack(image, label)
            adversarials.append(torch.from_numpy(adversarial).to(x))

        # TODO: Finire
        return utils.maybe_stack(adversarials, x.shape[1:], device=x.device)