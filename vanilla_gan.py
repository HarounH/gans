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
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--dlr', type=float, default=0.001, metavar='LR',
                    help='learning rate for discriminator (default: 0.001)')
parser.add_argument('--glr', type=float, default=0.001, metavar='LR',
                    help='learning rate for generator (default: 0.001)')
parser.add_argument('--betas', nargs=2,
                    type=float, default=(0.9, 0.999), help='Adam parameters')
# parser.add_argument('--d_noise_amp', default=0.001, type=float,
#                     help='Amplitude of noise added to input to discriminator during training')
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


# Model definition
class VanillaGAN(nn.Module):
    def __init__(self, xdim, gdims, ddims):
        super(VanillaGAN, self).__init__()
        generator = []
        current_dim = gdims[0]
        for i in range(1, len(gdims)):
            generator.append(nn.Linear(current_dim, gdims[i]))
            generator.append(nn.ReLU(True))
            current_dim = gdims[i]
        generator.append(nn.Linear(current_dim, xdim))
        generator.append(nn.Sigmoid())

        self.g = nn.Sequential(
            *generator
        )
        discriminator = []
        current_dim = xdim
        for i in range(len(ddims)):
            discriminator.append(nn.Linear(current_dim, ddims[i]))
            discriminator.append(nn.ReLU(True))
            # discriminator.append(nn.Dropout(dropout_prob))
            current_dim = ddims[i]
        discriminator.append(nn.Linear(current_dim, 1))
        discriminator.append(nn.Sigmoid())
        self.d = nn.Sequential(
            *discriminator
        )

    def gen(self, z):
        return self.g(z)

    def disc(self, x):
        return self.d(x)

    def forward(self, x, z):
        gen_x = self.gen(z)
        return gen_x, self.disc(gen_x), self.disc(x)


if __name__ == '__main__':
    xdim = int(np.prod(MNIST_SIZE))
    zdim = 100
    model = VanillaGAN(xdim, [zdim, 128], [128])  # , dropout_prob=args.dropout_prob)
    # import pdb; pdb.set_trace()
    for parameter in model.parameters():
        if len(parameter.size()) > 1:
            # Xavier init - fan out
            parameter.data.normal_(mean=0, std=np.sqrt(2.0 / parameter.size()[1]))
        else:
            parameter.data.zero_()  # bias terms
    # import pdb; pdb.set_trace()
    d_optim = optim.Adam(model.d.parameters(), lr=args.dlr, betas=args.betas)
    g_optim = optim.Adam(model.g.parameters(), lr=args.glr, betas=args.betas)
    train_loader = train_loader_maker(args.batch_size)
    test_loader = test_loader_maker(args.test_batch_size)
    # import pdb; pdb.set_trace()
    bce = F.binary_cross_entropy
    losses = []
    for epoch in range(args.epochs):
        # model.train()
        loss = {}
        loss['d_fake_loss'] = 0.0
        loss['d_real_loss'] = 0.0
        loss['d_loss'] = 0.0
        loss['g_loss'] = 0.0
        loss['count'] = 0
        for x, _ in train_loader:
            batch_size = x.size()[0]
            # Train!
            loss['count'] += batch_size
            x = x.view(batch_size, -1)
            z = torch.randn((batch_size, zdim))
            # import pdb; pdb.set_trace()
            g_z = model.g(z)
            d_fake = model.d(g_z)  # + args.d_noise_amp * torch.randn_like(g_z))
            d_real = model.d(x)  # + args.d_noise_amp * torch.randn_like(x))

            d_fake_loss = bce(d_fake, torch.zeros(batch_size, 1))
            d_real_loss = bce(d_real, torch.ones(batch_size, 1))
            d_loss = d_fake_loss + d_real_loss
            # g_loss = bce(d_fake, torch.ones(batch_size, 1))

            loss['d_fake_loss'] += d_fake_loss.item()
            loss['d_real_loss'] += d_real_loss.item()
            loss['d_loss'] += d_loss.item()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            z = torch.randn((batch_size, zdim))
            g_z = model.g(z)
            d_fake = model.d(g_z)  # + args.d_noise_amp * torch.randn_like(g_z))
            g_loss = bce(d_fake, torch.ones(batch_size, 1))
            loss['g_loss'] += g_loss.item()

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            del d_loss, g_loss
        loss['d_fake_loss'] /= loss['count']
        loss['d_real_loss'] /= loss['count']
        loss['d_loss'] /= loss['count']
        loss['g_loss'] /= loss['count']
        losses.append(loss)
        print('Epoch', epoch)
        print('d_fake_loss', losses[-1]['d_fake_loss'])
        print('d_real_loss', losses[-1]['d_real_loss'])
        print('d_loss', losses[-1]['d_loss'])
        print('g_loss', losses[-1]['g_loss'])
        if epoch % args.test_interval == 0:
            model.d.eval()
            fakes = []
            reals = []
            for x, _ in test_loader:
                # import pdb; pdb.set_trace()
                mb = x.shape[0]
                x = x.view(mb, -1)
                z = torch.randn((mb, zdim))
                gen_x, d_fake, d_real = model(x, z)
                fakes.append(bce(d_fake, torch.zeros(mb, 1)).item())
                reals.append(bce(d_real, torch.ones(mb, 1)).item())
            print('eval[' + str(epoch) + '] fake', sum(fakes) / len(fakes))
            print('eval[' + str(epoch) + '] real', sum(reals) / len(reals))
            # import pdb; pdb.set_trace()
            save_image(gen_x.view(-1, 28, 28).unsqueeze(1), 'images/vanilla_gan/generated_epoch' + str(epoch) + '.png', nrow=8)
            model.d.train()
