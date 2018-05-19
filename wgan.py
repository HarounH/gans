'''
    Implementing Wasserstein GAN
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


# argparse
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-tb', '--test_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--depochs', type=int, default=5, metavar='N',
                    help='number of epochs to train discriminator per main epoch (default: 5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--dlr', type=float, default=0.0001, metavar='LR',
                    help='learning rate for discriminator (default: 0.0001)')
parser.add_argument('--glr', type=float, default=0.0001, metavar='LR',
                    help='learning rate for generator (default: 0.0001)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1337, metavar='S',
                    help='random seed (default: 1337)')
parser.add_argument('--test_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before testing')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)


class WGAN(nn.Module):
    def __init__(self, xdim, zdim, gdims, ddims):
        super(WGAN, self).__init__()
        self.name = 'wgan'
        # Create stuff
        generator = []
        current_dim = zdim
        for dim in gdims:
            generator.append(nn.Linear(current_dim, dim))
            current_dim = dim
            generator.append(nn.ReLU(True))
        generator.append(nn.Linear(current_dim, xdim))
        generator.append(nn.Sigmoid())
        self.g = nn.Sequential(*generator)
        current_dim = xdim
        discriminator = []
        for dim in ddims:
            discriminator.append(nn.Linear(current_dim, dim))
            current_dim = dim
            discriminator.append(nn.ReLU(True))
        discriminator.append(nn.Linear(current_dim, 1))
        # discriminator.append(nn.Sigmoid())  ...
        self.d = nn.Sequential(*discriminator)

    def gen(self, z):
        return self.g(z)

    def disc(self, x):
        return self.d(x)

    def forward(self):
        return 0


if __name__ == '__main__':
    xdim = int(np.prod(MNIST_SIZE))
    zdim = 10
    # import pdb; pdb.set_trace()
    model = WGAN(xdim, zdim, [128], [128])

    for parameter in model.parameters():
        if len(parameter.size()) > 1:
            # Xavier init - fan out
            parameter.data.normal_(mean=0, std=np.sqrt(2.0 / parameter.size()[1]))
        else:
            parameter.data.zero_()  # bias terms

    d_optim = optim.Adam(model.d.parameters(), lr=args.dlr)
    g_optim = optim.Adam(model.g.parameters(), lr=args.glr)
    train_loader = train_loader_maker(args.batch_size)
    test_loader = test_loader_maker(args.test_batch_size)
    losses = []
    train_what = 0
    os.makedirs(os.path.join('images', model.name), exist_ok=True)
    for epoch in range(args.epochs):
        loss = {}
        loss['d_fake_loss'] = 0.0
        loss['d_real_loss'] = 0.0
        loss['d_loss'] = 0.0
        loss['g_loss'] = 0.0
        loss['dcount'] = 0
        loss['gcount'] = 0
        for x, _ in train_loader:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            z = torch.randn((batch_size, zdim))
            if train_what == args.depochs:  # Train generator
                loss['gcount'] += 1
                g_z = model.gen(z)
                d_fake = model.disc(g_z)
                g_loss = -torch.mean(d_fake)
                g_optim.zero_grad()
                g_loss.backward()
                g_optim.step()
                loss['g_loss'] += g_loss.item()
            else:  # Train discriminator
                loss['dcount'] += 1
                g_z = model.gen(z)
                d_fake = model.disc(g_z)
                d_real = model.disc(x)
                d_fake_loss = torch.mean(d_fake)
                d_real_loss = -torch.mean(d_real)
                d_loss = d_real_loss + d_fake_loss
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()
                for parameter in model.d.parameters():
                    parameter.data.clamp_(-0.01, 0.01)
                loss['d_fake_loss'] += d_fake_loss.item()
                loss['d_real_loss'] += d_real_loss.item()
                loss['d_loss'] += d_loss.item()
                # weight clipping?

            train_what = (train_what + 1) % (args.depochs + 1)
        loss['d_fake_loss'] /= loss['dcount']
        loss['d_real_loss'] /= loss['dcount']
        loss['d_loss'] /= loss['dcount']
        loss['g_loss'] /= loss['gcount']
        losses.append(loss)
        print('Epoch', epoch)
        print('d_fake_loss', losses[-1]['d_fake_loss'])
        print('d_real_loss', losses[-1]['d_real_loss'])
        print('d_loss', losses[-1]['d_loss'])
        print('g_loss', losses[-1]['g_loss'])

        if epoch % args.test_interval == 0:
            mb = args.test_batch_size
            z = torch.randn((mb, zdim))
            gen_x = model.gen(z)
            save_image(gen_x.view(-1, 28, 28).unsqueeze(1),
                       'images/' + model.name + '/generated_epoch' + str(epoch) + '.png',
                       nrow=int(np.sqrt(gen_x.shape[0])))
            print('Saved images on ', epoch, 'epoch')
