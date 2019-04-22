from __future__ import print_function
import argparse
import data
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functional import log_sum_exp
from torch.utils.data import DataLoader,TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False, help='CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--trainBatch', type = int, default = 64, help = 'training batch size')
parser.add_argument('--valBatch', type = int, default = 640, help = 'validation and test batch size')
parser.add_argument('--samplesClass', type = int, default = 10, help = 'samples per class for training')
parser.add_argument('--predict', action = 'store_true', help = 'evaluate model performance on test set')
parser.add_argument('--logInterval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--evalInterval', type=int, default=100, help='how many batches to wait before evaling training status')
parser.add_argument('--unlabelWeight', type=float, default=1, help='scale factor between labeled and unlabeled data')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--logDir', type=str, default='./logfile', help='logfile path, tensorboard format')
parser.add_argument('--saveDir', type=str, default='./models', help = 'saving path, pickle format')

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu')
print('device: {}'.format(device))

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

np.random.seed(args.seed)

# discriminator and generator models
class Discriminator(nn.Module):
    def __init__(self, nc = 1, ndf = 64, output_units = 10):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.main = nn.Sequential(
            # state size. (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 3, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 0, bias=False),
        )
        self.final = nn.Linear(ndf * 4, output_units, bias=False)
    def forward(self, x, feature = False):
        x_f = self.main(x).view(-1, self.ndf * 4)
        return x_f if feature else self.final(x_f)

class Generator(nn.Module):
    def __init__(self, z_dim, ngf = 64, output_dim = 28 ** 2):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf, 1, 4, 2, 3, bias=False),
            # state size. (ngf) x 32 x 32
            nn.Sigmoid()
        )
    def forward(self, batch_size):
        x = Variable(torch.rand(batch_size, self.z_dim, 1, 1), requires_grad = False, volatile = not self.training).to(device)
        return self.main(x)

def trainD(x_label, y, x_unlabel):
    # send data to device
    x_label, x_unlabel, y = Variable(x_label).to(device), Variable(x_unlabel).to(device), Variable(y, requires_grad = False).to(device)
    
    # pass data through discriminators
    output_label = netD(x_label)
    output_unlabel = netD(x_unlabel)
    output_fake_g = netG(x_unlabel.size()[0])
    output_fake_g = output_fake_g.view(x_unlabel.size()).detach()
    output_fake = netD(output_fake_g)

    # log ∑e^x_i
    logz_label = log_sum_exp(output_label)
    logz_unlabel = log_sum_exp(output_unlabel)
    logz_fake = log_sum_exp(output_fake)

    # log e^x_label = x_label 
    prob_label = torch.gather(output_label, 1, y.unsqueeze(1))

    # calculate the loss
    loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)
    loss_unsupervised = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                        torch.mean(F.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)
    loss = loss_supervised + args.unlabelWeight * loss_unsupervised
    
    # calculate acc
    acc = torch.mean((output_label.max(1)[1] == y).float())

    # zero out the optimizer
    optimD.zero_grad()

    # backpropagate loss
    loss.backward()

    # make a step with the discriminator optimizer
    optimD.step()
    return loss_supervised.data.cpu().numpy(), loss_unsupervised.data.cpu().numpy(), acc

def trainG(x_unlabel):
    fake = netG(x_unlabel.size()[0]).view(x_unlabel.size())
    mom_gen = netD(fake, feature=True)
    mom_unlabel = netD(Variable(x_unlabel).to(device), feature=True)
    mom_gen = torch.mean(mom_gen, dim = 0)
    mom_unlabel = torch.mean(mom_unlabel, dim = 0)
    loss_fm = torch.mean((mom_gen - mom_unlabel) ** 2)
    #loss_adv = -torch.mean(F.softplus(log_sum_exp(output_fake)))
    loss = loss_fm #+ 1. * loss_adv        
    
    # zero out the optimizer
    optimG.zero_grad()
    optimD.zero_grad()

    # backpropagate loss
    loss.backward()

    # make a step with the generator optimizer
    optimG.step()
    return loss.data.cpu().numpy()

def train():
    gn = 0

    for epoch in range(args.epochs):
        netG.train()
        netD.train()
        unlabeled_loader1 = DataLoader(unlabeled, batch_size = args.trainBatch, shuffle=True, drop_last=True, num_workers = 4)
        unlabeled_loader2_iter = iter(DataLoader(unlabeled, batch_size = args.trainBatch, shuffle=True, drop_last=True, num_workers = 4))
        labeled_loader = DataLoader(labeled, batch_size = args.trainBatch, shuffle=True, drop_last=True, num_workers = 4)
        labeled_loader_iter = iter(labeled_loader)
        loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
        batch_num = 0
        print("Total unlabeled batch", len(unlabeled_loader1))
        #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth'% (args.saveDir, epoch))
        #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.saveDir, epoch))
        for batch_idx, (unlabel1, _) in enumerate(unlabeled_loader1):
            unlabel1.to(device)

            batch_num += 1
            unlabel2, _ = unlabeled_loader2_iter.next()
            unlabel2.to(device)
            try:
                data, target = labeled_loader_iter.next()
            except StopIteration:
                labeled_loader_iter = iter(labeled_loader)
                data, target = labeled_loader_iter.next()
            data.to(device)
            target.to(device)

            # train discriminator
            ll, lu, acc = trainD(data, target, unlabel1)
            loss_supervised += ll
            loss_unsupervised += lu
            accuracy += acc

            # train generator
            lg = trainG(unlabel2)
            if epoch > 1 and lg > 1:
                lg = trainG(unlabel2)
            loss_gen += lg
            if (batch_num + 1) % args.logInterval == 0:
                print('Training: %d / %d' % (batch_num + 1, len(unlabeled_loader1)))
                gn += 1
                netD.train()
                netG.train()
        loss_supervised /= batch_num
        loss_unsupervised /= batch_num
        loss_gen /= batch_num
        accuracy /= batch_num
        print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))

        if (epoch + 1) % args.evalInterval == 0:
            print("Eval: correct %d / %d"  % (eval(), len(test_loader)))
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth'% (args.saveDir, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.saveDir, epoch))
                
def eval():
    netG.eval()
    netD.eval()
    num_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = netD(data)
            pred = output.data.max(1, keepdim=True)[1]
            num_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()  
    
    return num_correct
    
# create folders
if not os.path.exists(args.logDir):
    os.makedirs(args.logDir)
    
if not os.path.exists(args.saveDir):
    os.makedirs(args.saveDir)

# instantiate model
netG = Generator(100, ngf = 64, output_dim=28**2).to(device)
netD = Discriminator(nc = 1, output_units=10).to(device)
if args.netG != '':
    print('Loading Generator')
    netG.load_state_dict(torch.load(args.netG))
if args.netD != '':
    print('Loading Discriminator')
    netD.load_state_dict(torch.load(args.netD))

# instantiate dataloaders
labeled = data.get_labeled_train_data_tensor(None, args.samplesClass)
test_loader = data.get_labeled_test_data_loader(None, args.valBatch)
unlabeled = data.get_unlabeled_data(None)

# define loss function and optimizer
optimD = optim.Adam(netD.parameters(), lr=args.lr, betas= (args.momentum, 0.999))
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas = (args.momentum,0.999))

# train model
if not args.predict:
    print("Training my Super GAN")
    train()
else:
    print("Evaluating my Super GAN")
    eval()