import os

for es in ['5', '10', '20', '50', '100']:
    for arch in ['a', 'b', 'c']:
        os.system(f'python cli.py train-classifier cifar10 {arch} std:train 1000 trained-models/robust-classifiers/cifar10-{arch}-es{es}-ftr-bim0.1.pth --validation-split 0.1 --early-stopping {es} --flip --translation 0.1 --rotation 15 --adversarial-training bim --adversarial-ratio 0.5 --adversarial-p linf --adversarial-eps 0.1')