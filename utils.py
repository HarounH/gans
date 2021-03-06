import os
import sys
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


# Xavier initialization
def initialize_parameters(params):
    for parameter in params:
        if len(parameter.size()) > 1:
            # Xavier init - fan out
            parameter.data.normal_(mean=0, std=np.sqrt(2.0 / parameter.size()[1]))
        else:
            parameter.data.zero_()  # bias terms


def bce_loss(eps):
    def lossfn(input, target):
        return -torch.mean(target * torch.log(input + eps) + (1 - target) * torch.log(1 - input + eps))
    return lossfn


def mse_loss(x, y):
    return torch.mean(torch.sum((x - y)**2, 1))
