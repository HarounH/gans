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
parser.add_argument('--clr', type=float, default=0.001, metavar='LR',
                    help='learning rate for discriminator (default: 0.001)')
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


# Model
class InfoGAN(nn.Module):
    '''
        [zdim, cdim] -> xdim -> [zdim, cdim]
    '''
    def __init__(self, xdim, zdim, cdim, gdims, ddims, cdims):
        super(InfoGAN, self).__init__()
        self.name = 'info_gan'
        current_dim = zdim + cdim
        generator = []
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
        discriminator.append(nn.Sigmoid())
        self.d = nn.Sequential(*discriminator)

        c_pred_network = []
        current_dim = xdim
        for dim in cdims:
            c_pred_network.append(nn.Linear(current_dim, dim))
            current_dim = dim
            c_pred_network.append(nn.ReLU(True))
        c_pred_network.append(nn.Linear(current_dim, cdim))
        # c_pred_network.append(nn.Sigmoid())
        c_pred_network.append(nn.Softmax(dim=1))
        self.c = nn.Sequential(*c_pred_network)

    def gen(self, z, c):
        return self.g(torch.cat([z, c], dim=1))

    def disc(self, x):
        return self.d(x)

    def get_c(self, x):
        return self.c(x)


    def forward(self, x, z, c):
        raise NotImplementedError


if __name__ == '__main__':
    xdim = int(np.prod(MNIST_SIZE))
    zdim = 16
    cdim = 10
    model = InfoGAN(xdim, zdim, cdim, [128], [128], [128])

    for parameter in model.parameters():
        if len(parameter.size()) > 1:
            # Xavier init - fan out
            parameter.data.normal_(mean=0, std=np.sqrt(2.0 / parameter.size()[1]))
        else:
            parameter.data.zero_()  # bias terms

    d_optim = optim.Adam(model.d.parameters(), lr=args.dlr, betas=args.betas)
    g_optim = optim.Adam(model.g.parameters(), lr=args.glr, betas=args.betas)
    c_optim = optim.Adam(list(model.g.parameters()) + list(model.c.parameters()),
                         lr=args.clr,
                         betas=args.betas)

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
        loss['HQc'] = 0.0
        loss['Hc'] = 0.0
        loss['c_loss'] = 0.0
        loss['count'] = 0

        for x, _ in train_loader:
            batch_size = x.size()[0]
            # Train!
            loss['count'] += batch_size
            x = x.view(batch_size, -1)
            z = torch.randn((batch_size, zdim))
            c = torch.from_numpy(
                np.random.multinomial(
                    1,
                    cdim * [1.0 / cdim],
                    batch_size
                    )
                ).float()
            # discriminator training
            g_z = model.gen(z, c)
            d_fake = model.disc(g_z)  # + args.d_noise_amp * torch.randn_like(g_z))
            d_real = model.disc(x)  # + args.d_noise_amp * torch.randn_like(x))

            d_fake_loss = bce(d_fake, torch.zeros(batch_size, 1))
            d_real_loss = bce(d_real, torch.ones(batch_size, 1))
            # d_fake_loss = -torch.mean(torch.log(1 - d_fake + 1e-8))
            # d_real_loss = -torch.mean(torch.log(1e-8 + d_real))
            d_loss = d_fake_loss + d_real_loss
            # g_loss = bce(d_fake, torch.ones(batch_size, 1))

            loss['d_fake_loss'] += d_fake_loss.item()
            loss['d_real_loss'] += d_real_loss.item()
            loss['d_loss'] += d_loss.item()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # generator training
            # z = torch.randn((batch_size, zdim))
            g_z = model.gen(z, c)
            d_fake = model.disc(g_z)  # + args.d_noise_amp * torch.randn_like(g_z))
            g_loss = bce(d_fake, torch.ones(batch_size, 1))
            # g_loss = -torch.mean(torch.log(1e-8 + d_fake))
            loss['g_loss'] += g_loss.item()

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            # c_pred training
            g_z = model.gen(z, c)
            cpred = model.get_c(g_z)
            HQc = F.cross_entropy(cpred, torch.max(c, dim=1)[1])
            Hc = F.cross_entropy(c, torch.max(c, dim=1)[1])
            # HQc = torch.mean(-torch.sum(c * torch.log(cpred + 1e-8), dim=1))
            # Hc = torch.mean(-torch.sum(c * torch.log(c + 1e-8), dim=1))

            IQc = HQc + Hc
            loss['HQc'] += HQc.item()
            loss['Hc'] += Hc.item()
            loss['c_loss'] += IQc.item()
            c_optim.zero_grad()
            IQc.backward()
            c_optim.step()

            del HQc, Hc, IQc, d_fake_loss, d_real_loss, d_loss, g_loss
        loss['d_fake_loss'] /= loss['count']
        loss['d_real_loss'] /= loss['count']
        loss['d_loss'] /= loss['count']
        loss['g_loss'] /= loss['count']
        loss['c_loss'] /= loss['count']
        loss['HQc'] /= loss['count']
        loss['Hc'] /= loss['count']
        losses.append(loss)
        print('Epoch', epoch)
        print('d_fake_loss', losses[-1]['d_fake_loss'])
        print('d_real_loss', losses[-1]['d_real_loss'])
        print('d_loss', losses[-1]['d_loss'])
        print('Hc', losses[-1]['Hc'])
        print('HQc', losses[-1]['HQc'])
        print('c_loss', losses[-1]['c_loss'])
        print('g_loss', losses[-1]['g_loss'])
        if epoch % args.test_interval == 0:
            mb = args.test_batch_size
            z = torch.randn((mb, zdim))
            c = torch.from_numpy(
                np.random.multinomial(
                    1,
                    cdim * [1.0 / cdim],
                    mb
                    )
                ).float()
            gen_x = model.gen(z, c)
            os.makedirs(os.path.join('images', model.name), exist_ok=True)
            save_image(gen_x.view(-1, 28, 28).unsqueeze(1),
                       'images/' + model.name + '/generated_epoch' + str(epoch) + '.png',
                       nrow=int(np.sqrt(gen_x.shape[0])))
            print('Saved images on ', epoch, 'epoch')
            # model.d.train()
