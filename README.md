# Counter-Attack

Counter-Attack (also known as DL-VAX) is an adversarial metadefense that uses adversarial attacks to identify non-robust inputs.
It is a continuation of [counter-attack-legacy](https://github.com/samuelemarro/counter-attack-legacy).

# Installing

* Clone this repo
* `cd counter-attack`
* `pip install -r requirements.txt` or `conda install -r requirements.txt`
* For CUDA support, install a CUDA-compatible PyTorch version following the instructions [here](https://pytorch.org/get-started)
* For MIP support, install [MipVerify.jl v0.2.2](https://github.com/vtjeng/MIPVerify.jl) and [Gurobi](https://www.gurobi.com/)

# Running

For most actions, the entry point is `cli.py`. Use `python cli.py --help` for a list of commands.

# Examples

Train a CIFAR10 C classifier for 100 epochs with data augmentation and early stopping with patience = 10:
```python cli.py train-classifier mnist c std:train 100 /your/path/here.pth --flip --translation 0.1 --rotation 15 --early-stopping 10```

Train a MNIST A classifier for 200 epochs with PGD adversarial training and eps = 0.01:
```python cli.py train-classifier mnist a std:train 200 /your/path/here.pth --adversarial-training pgd --adversarial-p linf --adversarial-ratio 1 --adversarial-eps 0.01```

Train a CIFAR10 B classifier for 200 epochs with L1 and RS regularization (with RS minibatch = 32):
```python cli.py train-classifier mnist a std:train 200 /your/path/here.pth --l1-regularization 1e-5 --rs-regularization 1e-4 --rs-eps 0.01 --rs-minibatch 32```

Compute the accuracy of a CIFAR10 A model on the test set with batch size = 256:
```python cli.py accuracy cifar10 a std:test --state-dict-path /your/path/here.pth --batch-size 256```

Run a Carlini Linf attack against a MNIST C model and show the first 5 successful results:
```python cli.py attack mnist c std:test carlini linf --state-dict-path /your/path/here.pth --show 5```

Compare a Deepfool Linf attack with a BIM Linf attack against a CIFAR10 A model on genuines with index 500, 501, 502, 503 and 504:
```python cli.py compare cifar10 a std:test [bim, deepfool] linf --state-dict-path /your/path/here.pth --start 500 --stop 505```

Prune the weights of a MNIST B model with L1 magnitude less than 0.01:
```python cli.py prune-weights mnist b /original/path.pth /new/path.pth 0.01```

Prune the ReLUs of a CIFAR10 A model that are constant on 90% of the train dataset (with PGD Linf training and eps = 0.01):
```python cli.py prune-relu cifar10 a std:train pgd linf 1 0.01 /original/path.pth /new/path.pth 0.9```