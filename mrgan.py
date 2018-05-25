'''
    Trying to learn Mode Regularized GAN
'''

import os
import sys
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
# from tensorboardX import SummaryWriter
# Datasets!
from mnist import train_loader_maker, test_loader_maker, MNIST_SIZE
from utils import initialize_parameters, bce_loss, mse_loss

# argparse
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('-b', '--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-tb', '--test_batch_size', type=int, default=64,
                    metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=70, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--dlr', type=float, default=1e-4, metavar='LR',
                    help='learning rate for discriminator (default: 1e-4)')
parser.add_argument('--glr', type=float, default=1e-4, metavar='LR',
                    help='learning rate for generator (default: 1e-4)')
parser.add_argument('--elr', type=float, default=1e-4, metavar='LR',
                    help='learning rate for encoder (default: 1e-4)')
parser.add_argument('--lambdas', type=float, nargs=2, default=(1e-2, 1e-2),
                    help='lambdas from the MRGAN paper. Hyperparams for reg')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1337, metavar='S',
                    help='random seed (default: 1337)')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before testing')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)


class MRGAN(nn.Module):
    def __init__(self, xdim, zdim, gdims, ddims, edims):
        super(MRGAN, self).__init__()
        self.name = 'mrgan'
        self.mdgan = False  # Not yet implemented
        # Generator first
        gen = []
        current_dim = zdim
        for dim in gdims:
            gen.append(nn.Linear(current_dim, dim))
            gen.append(nn.ReLU(True))
            current_dim = dim
        gen.append(nn.Linear(current_dim, xdim))
        gen.append(nn.Sigmoid())
        self.g = nn.Sequential(*gen)
        # Encoder second
        enc = []
        current_dim = xdim
        for dim in edims:
            enc.append(nn.Linear(current_dim, dim))
            enc.append(nn.ReLU(True))
            current_dim = dim
        enc.append(nn.Linear(current_dim, zdim))
        self.e = nn.Sequential(*enc)
        # discriminator third
        disc = []
        current_dim = xdim
        for dim in ddims:
            disc.append(nn.Linear(current_dim, dim))
            disc.append(nn.ReLU(True))
            current_dim = dim
        disc.append(nn.Linear(current_dim, 1))
        disc.append(nn.Sigmoid())
        self.d1 = nn.Sequential(*disc)
        if self.mdgan:
            disc = []
            current_dim = xdim
            for dim in ddims:
                disc.append(nn.Linear(current_dim, dim))
                disc.append(nn.ReLU(True))
                current_dim = dim
            disc.append(nn.Linear(current_dim, 1))
            disc.append(nn.Sigmoid())
            self.d2 = nn.Sequential(*disc)

    def gen(self, z):
        return self.g(z)

    def enc(self, x):
        return self.e(x)

    def disc1(self, x):
        return self.d1(x)

    def disc2(self, x):
        return self.d2(x)

    def forward(self):
        raise NotImplementedError


if __name__ == '__main__':
    xdim = int(np.prod(MNIST_SIZE))
    zdim = 128
    model = MRGAN(xdim, zdim, [128], [128], [128])
    # initialize_parameters(model.parameters())
    e_optim = optim.Adam(model.e.parameters(), lr=args.elr)
    d1_optim = optim.Adam(model.d1.parameters(), lr=args.dlr)
    if model.mdgan:
        d2_optim = optim.Adam(model.d2.parameters(), lr=args.dlr)
    g_optim = optim.Adam(model.g.parameters(), lr=args.glr)

    train_loader = train_loader_maker(args.batch_size)
    test_loader = test_loader_maker(args.test_batch_size)

    bce = bce_loss(1e-8)  # F.binary_cross_entropy
    mse = mse_loss  # F.mse_loss
    losses = []

    for epoch in range(args.epochs):
        # Training
        loss = {}
        loss['d1_loss'] = 0.0
        loss['d2_loss'] = 0.0
        loss['e_loss'] = 0.0
        loss['g_loss'] = 0.0
        loss['count'] = 0
        for x, _ in train_loader:
            mb = x.shape[0]
            x = x.view(mb, -1)
            # Manifold step
            ## discriminator 1
            z = torch.randn((mb, zdim))
            regen_x = model.gen(z)
            d_fake = model.disc1(regen_x)
            d_real = model.disc1(x)
            d1_fake_loss = bce(d_fake, torch.zeros(mb, 1))  # -torch.mean(torch.log(1 - d_fake + 1e-8))  #
            d1_real_loss = bce(d_real, torch.ones(mb, 1))  # -torch.mean(torch.log(d_real + 1e-8))  #
            d1_loss = d1_fake_loss + d1_real_loss
            loss['d1_loss'] += d1_loss.item()
            d1_optim.zero_grad()
            d1_loss.backward()
            d1_optim.step()

            ## generator
            z = torch.randn((mb, zdim))
            gen_z = model.gen(z)
            d1_fake = model.disc1(gen_z)

            regen_x = model.gen(model.enc(x))
            d1_regen = model.disc1(regen_x)

            mse_loss_val = torch.sum((regen_x-x)**2, 1)
            d1_regen_loss = bce(d1_regen, torch.ones(mb, 1))  # torch.log(d1_regen + 1e-8)  #

            g_loss = bce(d1_fake, torch.ones(mb, 1)) + \
                torch.mean(
                    args.lambdas[0] * mse_loss_val +
                    args.lambdas[1] * d1_regen_loss
                    )
            loss['g_loss'] += g_loss.item()
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # Encoder
            regen_x = model.gen(model.enc(x))
            d1_regen = model.disc1(regen_x)

            mse_loss_val = torch.sum((regen_x-x)**2, 1)
            d1_regen_loss = bce(d1_regen, torch.ones(mb, 1))  # torch.log(d1_regen + 1e-8)  # 
            e_loss = torch.mean(
                args.lambdas[0] * mse_loss_val +
                args.lambdas[1] * d1_regen_loss
                )
            loss['e_loss'] += e_loss.item()
            e_optim.zero_grad()
            e_loss.backward()
            e_optim.step()

            if model.mdgan:
                raise NotImplementedError
                # Diffusion step
                ## discriminator 2
                z = torch.randn((mb, zdim))
                gen_z = model.gen(z)
                regen_x = model.gen(model.enc(x))
                d2_gen = model.disc2(gen_z)
                d2_regen = model.disc2(regen_x)
                d2_gen_loss = bce(d2_gen, torch.ones(mb, 1))
                d2_regen_loss = bce(d2_regen, torch.zeros(mb, 1))
                d2_loss = d2_gen_loss + d2_regen_loss
                d2_optim.zero_grad()
                d2_loss.backward()
                d2_optim.step()
                # encoder
                # generator
            loss['count'] += 1
        loss['d1_loss'] /= loss['count']
        loss['d2_loss'] /= loss['count']
        loss['g_loss'] /= loss['count']
        loss['e_loss'] /= loss['count']
        print('Epoch', epoch)
        for key, value in loss.items():
            print('\t', key, epoch, value)
        losses.append(loss)
        # Testing
        if epoch % args.test_interval == 0:
            mb = args.test_batch_size
            z = torch.randn((mb, zdim))
            gen_x = model.gen(z)
            os.makedirs(os.path.join('images', model.name), exist_ok=True)
            save_image(gen_x.view(-1, 28, 28).unsqueeze(1),
                       'images/' + model.name + '/generated_epoch' + str(epoch) + '.png',
                       nrow=int(np.sqrt(gen_x.shape[0])))
            print('Saved images on ', epoch, 'epoch')
