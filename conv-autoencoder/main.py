# import dependencies
import argparse
import numpy as np
import os
import tqdm

import torch
import torch.nn as nn 
import torch.optim as optim

import data
import dnn

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type = int, default = 128, help = 'training batch size')
parser.add_argument('-d', '--data', type = str, default = os.getcwd(), help = 'absolute path to datasets')
parser.add_argument('-e', '--epochs', type = int, default = 30, help = 'total number of training epochs')
parser.add_argument('-l', '--load', action = 'store_true', help = 'load pre-trained model parameters')
parser.add_argument('-m', '--model', type = str, default = os.getcwd(), help = 'absolute path to model parameters')

args = parser.parse_args()

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device, '\n')

# load data and instantiate dataloader
dataloader = data.load_dataset(args.data, training_batch_size = args.batch_size)

# instantiate neural net
net = dnn.Net().to(device)

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

# train neural net
for epoch in range(args.epochs):
    print('epoch', epoch + 1, 'of', args.epochs)
    
    batch_idx = None
    running_loss = 0

    for batch_idx, data in enumerate(tqdm.tqdm(dataloader)):
        inputs, _ = data
        inputs = inputs.to(device)

        # optimize
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # print epoch end loss
    print('training loss:', running_loss/(batch_idx + 1))