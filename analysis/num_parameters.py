import sys


sys.path.append('.')

import json
from pathlib import Path

import click
import numpy as np
import scipy.stats
import torch

import parsing

@click.command()
@click.argument('domain')
def main(domain):
    for architecture in ['a', 'b', 'c']:
        model = parsing.parse_model(domain, architecture, None, False, False, True)
        num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(num_params)

if __name__ == '__main__':
    main()