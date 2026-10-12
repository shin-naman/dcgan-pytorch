"""Configure reproducibility and select the training device."""

import random

import torch


def configure_runtime(args):
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)

    print(args)
    print('Random Seed: ', args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.use_deterministic_algorithms(True)

    # --ngpu 0 explicitly forces CPU; otherwise prefer CUDA, then MPS.
    if args.ngpu == 0:
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')
    return device
