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
parser.add_argument('-d', '--data', type = str, default = os.getcwd(), help = 'absolute path to dataset')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False,
                        help='CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('-l', '--load', action = 'store_true', help = 'load pre-trained model')
parser.add_argument('-s', '--split', type = float, default = 0.8, help = 'training and validation split ratio')
parser.add_argument('-t', '--train_batch', type = int, default = 128, help = 'training batch size')
parser.add_argument('-v', '--val_batch', type = int, default = 1, help = 'validation and test batch size')
parser.add_argument('--samples_class', type = int, default = 64, help = 'samples per class for training')
parser.add_argument('-p', '--predict', action = 'store_true', help = 'evaluate model performance on test set')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before evaling training status')
parser.add_argument('--unlabel-weight', type=float, default=1, metavar='N',
                        help='scale factor between labeled and unlabeled data')
parser.add_argument('--logdir', type=str, default='./logfile', metavar='LOG_PATH', help='logfile path, tensorboard format')
parser.add_argument('--savedir', type=str, default='./models', metavar='SAVE_PATH', help = 'saving path, pickle format')

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu')
print('device: {}\n'.format(device))

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

np.random.seed(args.seed)

# discriminator and generator models
class Discriminator(nn.Module):
    def __init__(self, nc = 1, ndf = 64, output_units = 1000):
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
        #print(output_dim)
        #print(ngf)
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
            nn.ConvTranspose2d(ngf, 3, 4, 2, 3, bias=False),
            # state size. (ngf) x 32 x 32
            nn.Sigmoid()
        )
    def forward(self, batch_size):
        x = Variable(torch.rand(batch_size, self.z_dim, 1, 1), requires_grad = False, volatile = not self.training).to(device)
        #if cuda:
        #    x = x.cuda()
        return self.main(x)

def trainD(x_label, y, x_unlabel):
    # send data to device
    x_label, x_unlabel, y = Variable(x_label).to(device), Variable(x_unlabel).to(device), Variable(y, requires_grad = False).to(device)
    
    # pass data through discriminators
    #output_label, output_unlabel, output_fake = netD(x_label), netD(x_unlabel), netD(netG(x_unlabel.size()[0]).view(x_unlabel.size()).detach())
    output_label = netD(x_label)
    output_unlabel = netD(x_unlabel)
    #print(x_unlabel.size())
    #print(x_unlabel.size()[0])
    output_fake_g = netG(x_unlabel.size()[0])
    #print(output_fake_g.size())
    output_fake_g = output_fake_g.view(x_unlabel.size()).detach()
    output_fake = netD(output_fake_g)
    #output_fake = netD(netG(x_unlabel.size()[0]).view(x_unlabel.size()).detach())

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
    loss = loss_supervised + args.unlabel_weight * loss_unsupervised
    
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
#        fake.retain_grad()
    #mom_gen, output_fake = netD(fake, feature=True)
    mom_gen = netD(fake, feature=True)
    #mom_unlabel, _ = netD(Variable(x_unlabel).to(device), feature=True)
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
    #assert self.unlabeled.__len__() > self.labeled.__len__()
    #assert type(self.labeled) == TensorDataset
    #times = int(np.ceil(len(unlabeled) * 1. / len(labeled_loader)))
    #print("times", times)
    #t1 = labeled.tensors[0].clone()
    #t2 = labeled.tensors[1].clone()
    #tile_labeled = TensorDataset(t1.repeat(times,1,1,1),t2.repeat(times))
    gn = 0


    for epoch in range(args.epochs):
        netG.train()
        netD.train()
        unlabeled_loader1 = DataLoader(unlabeled, batch_size = args.batch_size, shuffle=True, drop_last=True, num_workers = 4)
        unlabeled_loader2_iter = iter(DataLoader(unlabeled, batch_size = args.batch_size, shuffle=True, drop_last=True, num_workers = 4)) #.__iter__()
        labeled_loader = DataLoader(labeled, batch_size = args.batch_size, shuffle=True, drop_last=True, num_workers = 4)
        labeled_loader_iter = iter(labeled_loader)
        loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
        batch_num = 0
        print("Total unlabeled batch", len(unlabeled_loader1))
        for batch_idx, (unlabel1, _) in enumerate(unlabeled_loader1):
            print(batch_idx)
            unlabel1.to(device)
            #rint(unlabel1)
            #print(_label1)
# pdb.set_trace()
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

            #print(data)
            #print(target)

            # train discriminator
            ll, lu, acc = trainD(data, target, unlabel1)
            loss_supervised += ll
            loss_unsupervised += lu
            accuracy += acc

            # train generator
            lg = trainG(unlabel2)
            if epoch > 1 and lg > 1:
#                    pdb.set_trace()
                lg = trainG(unlabel2)
            loss_gen += lg
            if (batch_num + 1) % args.log_interval == 0:
                print('Training: %d / %d' % (batch_num + 1, len(unlabeled_loader1)))
                gn += 1
                netD.train()
                netG.train()
        loss_supervised /= batch_num
        loss_unsupervised /= batch_num
        loss_gen /= batch_num
        accuracy /= batch_num
        print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))
        sys.stdout.flush()
        if (epoch + 1) % args.eval_interval == 0:
            print("Eval: correct %d / %d"  % (eval(), len(test_loader)))
            torch.save(netG, os.path.join(args.savedir, 'netG.pkl'))
            torch.save(netD, os.path.join(args.savedir, 'netD.pkl'))
                
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
    """
        d, l = [], []
        for (datum, label) in self.test:
            d.append(datum)
            l.append(label)
        x, y = torch.stack(d), torch.LongTensor(l)
        x = Variable(x).to(device)
        y = Variable(y).to(device)
        pred = torch.max(netD(x), 1)[1].data
        s = torch.sum(pred == y)
        return s
    """
    

# instantiate model
if args.load:
    print('loading pre-trained model\n')
    netG = torch.load(os.path.join(args.savedir, 'netG.pkl'))
    netD = torch.load(os.path.join(args.savedir, 'netD.pkl'))
else:
    netG = Generator(100, ngf = 64, output_dim=96**2).to(device)
    netD = Discriminator(nc = 3, output_units=1000).to(device)

# instantiate dataloaders
labeled = data.get_labeled_train_data_tensor(args.data, args.samples_class)
#labeled_loader = data.get_labeled_train_data_loader(args.data, args.train_batch)
test_loader = data.get_labeled_test_data_loader(args.data, args.val_batch)
unlabeled = data.get_unlabeled_data(args.data)

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

            
"""
class ImprovedGAN(object):
    def __init__(self, G, D, labeled, unlabeled, test, args):
        if os.path.exists(args.savedir):
            print('Loading model from ' + args.savedir)
            self.G = torch.load(os.path.join(args.savedir, 'G.pkl'))
            self.D = torch.load(os.path.join(args.savedir, 'D.pkl'))
        else:
            os.makedirs(args.savedir)
            self.G = G
            self.D = D
            torch.save(self.G, os.path.join(args.savedir, 'G.pkl'))
            torch.save(self.D, os.path.join(args.savedir, 'D.pkl'))
        #self.writer = tensorboardX.SummaryWriter(log_dir=args.logdir)
        #if args.cuda:
        self.G.to(device)
        self.D.to(device)
        self.labeled = labeled
        self.unlabeled = unlabeled
        self.test = test
        self.Doptim = optim.Adam(self.D.parameters(), lr=args.lr, betas= (args.momentum, 0.999))
        self.Goptim = optim.Adam(self.G.parameters(), lr=args.lr, betas = (args.momentum,0.999))
        self.args = args
        #self.device = torch.device("cuda:0" if opt.cuda else "cpu")
    def trainD(self, x_label, y, x_unlabel):
        x_label, x_unlabel, y = Variable(x_label).to(device), Variable(x_unlabel).to(device), Variable(y, requires_grad = False).to(device)
        #if self.args.cuda:
        #    x_label, x_unlabel, y = x_label.cuda(), x_unlabel.cuda(), y.cuda()
        output_label, output_unlabel, output_fake = self.D(x_label, cuda=self.args.cuda), self.D(x_unlabel, cuda=self.args.cuda), self.D(self.G(x_unlabel.size()[0], cuda = self.args.cuda).view(x_unlabel.size()).detach(), cuda=self.args.cuda)
        logz_label, logz_unlabel, logz_fake = log_sum_exp(output_label), log_sum_exp(output_unlabel), log_sum_exp(output_fake) # log ∑e^x_i
        prob_label = torch.gather(output_label, 1, y.unsqueeze(1)) # log e^x_label = x_label 
        loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)
        loss_unsupervised = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                            torch.mean(F.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)
        loss = loss_supervised + self.args.unlabel_weight * loss_unsupervised
        acc = torch.mean((output_label.max(1)[1] == y).float())
        self.Doptim.zero_grad()
        loss.backward()
        self.Doptim.step()
        return loss_supervised.data.cpu().numpy(), loss_unsupervised.data.cpu().numpy(), acc
    
    def trainG(self, x_unlabel):
        fake = self.G(x_unlabel.size()[0], cuda = self.args.cuda).view(x_unlabel.size())
#        fake.retain_grad()
        mom_gen, output_fake = self.D(fake, feature=True, cuda=self.args.cuda)
        mom_unlabel, _ = self.D(Variable(x_unlabel).to(device), feature=True, cuda=self.args.cuda)
        mom_gen = torch.mean(mom_gen, dim = 0)
        mom_unlabel = torch.mean(mom_unlabel, dim = 0)
        loss_fm = torch.mean((mom_gen - mom_unlabel) ** 2)
        #loss_adv = -torch.mean(F.softplus(log_sum_exp(output_fake)))
        loss = loss_fm #+ 1. * loss_adv        
        self.Goptim.zero_grad()
        self.Doptim.zero_grad()
        loss.backward()
        self.Goptim.step()
        return loss.data.cpu().numpy()

    def train(self):
        assert self.unlabeled.__len__() > self.labeled.__len__()
        assert type(self.labeled) == TensorDataset
        times = int(np.ceil(self.unlabeled.__len__() * 1. / self.labeled.__len__()))
        t1 = self.labeled.tensors[0].clone()
        t2 = self.labeled.tensors[1].clone()
        tile_labeled = TensorDataset(t1.repeat(times,1,1,1),t2.repeat(times))
        gn = 0
        for epoch in range(self.args.epochs):
            self.G.train()
            self.D.train()
            unlabel_loader1 = DataLoader(self.unlabeled, batch_size = self.args.batch_size, shuffle=True, drop_last=True, num_workers = 4)
            unlabel_loader2 = DataLoader(self.unlabeled, batch_size = self.args.batch_size, shuffle=True, drop_last=True, num_workers = 4).__iter__()
            label_loader = DataLoader(tile_labeled, batch_size = self.args.batch_size, shuffle=True, drop_last=True, num_workers = 4).__iter__()
            loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
            batch_num = 0
            for (unlabel1, _label1) in unlabel_loader1:
#                pdb.set_trace()
                batch_num += 1
                unlabel2, _label2 = unlabel_loader2.next()
                x, y = label_loader.next()
                x.to(device)
                y.to(device)
                #if args.cuda:
                #    x, y, unlabel1, unlabel2 = x.cuda(), y.cuda(), unlabel1.cuda(), unlabel2.cuda()
                ll, lu, acc = self.trainD(x, y, unlabel1)
                loss_supervised += ll
                loss_unsupervised += lu
                accuracy += acc
                lg = self.trainG(unlabel2)
                if epoch > 1 and lg > 1:
#                    pdb.set_trace()
                    lg = self.trainG(unlabel2)
                loss_gen += lg
                if (batch_num + 1) % self.args.log_interval == 0:
                    print('Training: %d / %d' % (batch_num + 1, len(unlabel_loader1)))
                    gn += 1
                    #self.writer.add_scalars('loss', {'loss_supervised':ll, 'loss_unsupervised':lu, 'loss_gen':lg}, gn)
                    #self.writer.add_histogram('real_feature', self.D(Variable(x, volatile = True), cuda=self.args.cuda, feature = True)[0], gn)
                    #self.writer.add_histogram('fake_feature', self.D(self.G(self.args.batch_size, cuda = self.args.cuda), cuda=self.args.cuda, feature = True)[0], gn)
                    #self.writer.add_histogram('fc3_bias', self.G.fc3.bias, gn)
                    #self.writer.add_histogram('D_feature_weight', self.D.layers[-1].weight, gn)
#                    self.writer.add_histogram('D_feature_bias', self.D.layers[-1].bias, gn)
                    #print('Eval: correct %d/%d, %.4f' % (self.eval(), self.test.__len__(), acc))
                    self.D.train()
                    self.G.train()
            loss_supervised /= batch_num
            loss_unsupervised /= batch_num
            loss_gen /= batch_num
            accuracy /= batch_num
            print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))
            sys.stdout.flush()
            if (epoch + 1) % self.args.eval_interval == 0:
                print("Eval: correct %d / %d"  % (self.eval(), self.test.__len__()))
                torch.save(self.G, os.path.join(args.savedir, 'G.pkl'))
                torch.save(self.D, os.path.join(args.savedir, 'D.pkl'))
                

    #def predict(self, x):
    #    return torch.max(self.D(Variable(x).to(device), cuda=self.args.cuda), 1)[1].data
    def eval(self):
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            d, l = [], []
            for (datum, label) in self.test:
                d.append(datum)
                l.append(label)
            x, y = torch.stack(d), torch.LongTensor(l)
            x = Variable(x).to(device)
            y = Variable(y).to(device)
            pred = torch.max(self.D(x, cuda=self.args.cuda), 1)[1].data
            s = torch.sum(pred == y)
            return s
    
    
    def draw(self, batch_size):
        self.G.eval()
        return self.G(batch_size, cuda=self.args.cuda)
    """

