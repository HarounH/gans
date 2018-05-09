import os
import sys
import pdb
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

MNIST_SIZE = (28, 28)
# Dataset - MNIST
train_loader_maker = lambda batch_size: torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       # transforms.ToPILImage(),
                       transforms.ToTensor()#,
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader_maker = lambda test_batch_size: torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor()#,
        # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)
