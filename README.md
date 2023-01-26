# Counter-Attack

Counter-Attack is an adversarial metadefense that uses adversarial attacks to identify non-robust inputs.
This repo is the companion code of the paper "Computational Asymmetries in Robust Classification".

# Installing

* Clone this repo
* `cd counter-attack`
* `pip install -r requirements.txt` or `conda install -r requirements.txt`
* For CUDA support, install a CUDA-compatible PyTorch version following the instructions [here](https://pytorch.org/get-started)
* For MIP support, install [MipVerify.jl v0.2.2](https://github.com/vtjeng/MIPVerify.jl) and [Gurobi](https://www.gurobi.com/). All package versions are reported in version_info.txt

# Running

For most actions, the entry point is `cli.py`. Use `python cli.py --help` for a list of commands.

## Analysis

In addition to the main entry point, we provide a series of analysis scripts in ./analysis.
They can be run on their own from the root folder, e.g.:
```
python analysis/dataset_breakdown.py --help
```

Note that most analysis scripts automatically save their results in a given folder.

## HPC Scripts

We also include the Python scripts that were designed to be run in an HPC cluster. They can be found in ./batch_processing. The HPC scripts provide an all-in-one management workflow for training, attacking and collecting the results. Similarly to analysis scripts, they can be run on their own.

# Results

## Pretrained Weights

The pretrained weights are available anonymously [here](https://figshare.com/s/09524a278aba9cbb7be0). Most scripts will expect to find the weights for standard and adversarial models in trained-models/classifiers/[standard, adversarial]/, while for ReLU they are usually in trained-models/classifiers/relu/relu-pruned. Note that ReLU models use ReLU masking and therefore often need to be loaded with the `--masked-relu` option.

## UG100 Adversarial Examples

The adversarial examples are available anonymously at the following links:
- [MIP](https://figshare.com/s/ae397c93470b4ac80bb4)
- [Heuristic Strong](https://figshare.com/s/b4841d9fa2d8e79af1d7)
- [Heuristic Balanced](https://figshare.com/s/17130277649b015f492d)

The datasets can be loaded as follows:
```
import utils
dataset = utils.load_zip('/path/to/the/dataset.zip')
```

## MNIST and CIFAR10 Indices

The chosen indices for UG100 are reported in mnist_indices_intersection.json and cifar10_indices_intersection.json.

## UG100 Distances
For ease of study, we also include the distances of the adversarial examples in JSON format. They can be found in ./analysis/distances.

# Attack Parameter Sets
We provide three attack parameter sets:
- original_mip_attack_configuration.cfg, which was used as a warm-up heuristic for MIPVerify
- default_attack_configuration.cfg, which is referred to in the paper as the "strong" attack set.
The default and original_mip sets are equivalent for CIFAR10
- balanced_attack_configuration.cfg, which is a simplified version of the default attack set.

Attack configuration files follow a "specific-beats-generic" logic: if there is a parameter provided for both MNIST and MNIST Linf, the MNIST Linf value is chosen (as long as we are attacking a MNIST model with a Linf attack).

# Examples

## Training and Attacking

Train a CIFAR10 C classifier for 100 epochs with data augmentation and early stopping with patience = 10:
```
python cli.py train-classifier mnist c std:train 100 /your/path/here.pth --flip --translation 0.1 --rotation 15 --early-stopping 10
```

Train a MNIST A classifier for 200 epochs with PGD adversarial training and eps = 0.01:
```
python cli.py train-classifier mnist a std:train 200 /your/path/here.pth --adversarial-training pgd --adversarial-p linf --adversarial-ratio 1 --adversarial-eps 0.01
```

Train a CIFAR10 B classifier for 200 epochs with L1 and RS regularization (with RS minibatch = 32):
```
python cli.py train-classifier mnist a std:train 200 /your/path/here.pth --l1-regularization 1e-5 --rs-regularization 1e-4 --rs-eps 0.01 --rs-minibatch 32
```

Compute the accuracy of a CIFAR10 A model on the test set with batch size = 256:
```
python cli.py accuracy cifar10 a std:test --state-dict-path /your/path/here.pth --batch-size 256
```

Run a Carlini Linf attack against a MNIST C model and show the first 5 successful results:
```
python cli.py attack mnist c std:test carlini linf --state-dict-path /your/path/here.pth --show 5
```

Compare a Deepfool Linf attack with a BIM Linf attack against a CIFAR10 A model on genuines with index 500, 501, 502, 503 and 504:
```
python cli.py compare cifar10 a std:test [bim, deepfool] linf --state-dict-path /your/path/here.pth --start 500 --stop 505
```

Prune the weights of a MNIST B model with L1 magnitude less than 0.01:
```
python cli.py prune-weights mnist b /original/path.pth /new/path.pth 0.01
```

Prune the ReLUs of a CIFAR10 A model that are constant on 90% of the train dataset (with PGD Linf training and eps = 0.01):
```
python cli.py prune-relu cifar10 a std:train pgd linf 1 0.01 /original/path.pth /new/path.pth 0.9
```

## HPC Processing

Train and prune a MNIST C model with ReLU training:
```
python batch_processing/train_and_prune.py mnist c relu
```

Run both heuristic and MIP attacks on the CIFAR10 A model (trained with adversarial training), using the original_mip set, on indices from 100 (included) to 200 (excluded):
```
python batch_processing/attack_and_mip.py cifar10 a adversarial original 100 200
```

## Analysis

Determine the best pool of a given size for the MNIST balanced set (convergence atol=1e-5, rtol=1e-10):
```
python analysis/best_pool_by_size.py mnist balanced 1e-5 1e-10
```

Determine the composition of the chosen MNIST indices:
```
python analysis/dataset_breakdown.py mnist
```

Perform quantile calibration on CIFAR10 strong (convergence atol=1e-5, rtol=1e-10, folds = 5)
```
python analysis/quantile_calibration.py cifar10 strong 1e-5 1e-10 5
```

# Licenses

The license for our code can be found in ./LICENSE. We also report the licenses for the three used datasets (MNIST, CIFAR10 and UG100, our original dataset) in ./dataset_licenses.
