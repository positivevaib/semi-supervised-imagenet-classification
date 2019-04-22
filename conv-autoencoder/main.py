# import dependencies
import argparse
import os

import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
import tqdm
import visdom

import data
import model

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type = str, default = None, help = 'absolute path to model checkpoint')
parser.add_argument('--data', type = str, default = None, help = 'absolute path to datasets')
parser.add_argument('--epochs', type = int, default = 30, help = 'total number of training epochs')
parser.add_argument('--file', type = str, default = None, help = 'absolute path to file to dump stdout')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'initial learning rate')
parser.add_argument('--load', action = 'store_true', help = 'load pre-trained model')
parser.add_argument('--model', type = str, default = None, help = 'absolute path to save or load model')
parser.add_argument('--predict', action = 'store_true', help = 'evaluate model performance on test set')
parser.add_argument('--resume', action = 'store_true', help = 'resume training from checkpoint')
parser.add_argument('--split', type = float, default = 0.8, help = 'training and validation split ratio')
parser.add_argument('--train_batch', type = int, default = 128, help = 'training batch size')
parser.add_argument('--val_batch', type = int, default = 640, help = 'validation and test batch size')

args = parser.parse_args()

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: {}\n'.format(device))

# instantiate dataloaders
train_loader, val_loader = data.get_data_loaders(args.data, train_ratio = args.split, train_batch_size = args.train_batch, val_batch_size = args.val_batch)

print('training set: {}'.format(len(train_loader) * args.train_batch))
print('validation set: {}\n'.format(len(val_loader) * args.val_batch))

# instantiate model
if args.load:
    print('loading pre-trained model\n')
    model = torch.load(args.model, map_location = device)
else:
    model = model.Model().to(device)

# define loss function, optimizer and learning rate scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 0, verbose = True)

# load model checkpoint
if args.resume:
    model = torch.load(args.checkpoint['model_state_dict'], map_location = device)
    optimizer = torch.load(args.checkpoint['optimizer_state_dict'], map_location = device)
    scheduler = torch.load(args.checkpoint['scheduler_state_dict'], map_location = device)

# train model
if not args.predict:
    print('training model\n')

    # setup visdom for loss history visualization
    viz = visdom.Visdom()
    env_name = 'semi-supervised-imagenet-classification'
    plot = None

    # load checkpoints, if training to be resumed
    if args.resume:
        best_val_loss = args.checkpoint['best_val_loss']
        no_improvement = args.checkpoint['no_improvement']
        start_epoch = args.checkpoint['epoch']
    else:
        best_val_loss = float('inf')
        no_improvement = 0
        start_epoch = 0        

    for epoch in range(start_epochs, args.epochs):
        # setup progress bar
        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm.tqdm(desc = desc.format(0), total = len(train_loader), leave = False, file = args.file, initial = 0)

        tqdm.tqdm.write('epoch {} of {}'.format(epoch + 1, args.epochs), file = args.file)

        batch_idx = None
        running_loss = 0
        for batch_idx, data in enumerate(train_loader):
            inputs, _ = data
            inputs = inputs.to(device)

            # optimize
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # plot loss history
            if not plot:
                plot = viz.line(X = np.array([((epoch * len(train_loader)) + batch_idx + 1), ((epoch * len(train_loader)) + batch_idx + 1)]), 
                                Y = np.array([loss.item(), loss.item()]), env = env_name, 
                                opts = dict(legend = ['train'], title = 'loss hist.', xlabel = 'iters.', ylabel = 'loss'))
            else:
                viz.line(X = np.array([((epoch * len(train_loader)) + batch_idx + 1), ((epoch * len(train_loader)) + batch_idx + 1)]), 
                        Y = np.array([loss.item(), loss.item()]), env = env_name, win = plot, name = 'train', update = 'append')

            # print initial train loss
            if batch_idx == 0:
                tqdm.tqdm.write('initial training loss: {:.2f}'.format(loss.item()), file = args.file)

        # evaluate model performance on val set
        model.eval()
        with torch.no_grad():
            running_val_loss = 0
            for val_batch_idx, val_data in enumerate(val_loader):
                val_inputs, _ = val_data
                val_inputs = val_inputs.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_inputs)

                running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss / (val_batch_idx + 1)

            # invoke learning rate scheduler
            scheduler.step(metrics = avg_val_loss)
        
        model.train()

            # update progress bar
            pbar.desc = desc.format(loss.item())
            pbar.update()
        
        # print epoch end loss
        tqdm.tqdm.write('training loss: {:.2f}, validation loss: {:.2f}\n'.format(running_loss / (batch_idx + 1), avg_val_loss), 
                        file = args.file)

        # close progress bar
        pbar.close()

        # save checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': mode.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict(), 'best_val_loss': best_val_loss, 'no_improvement': no_improvement})

        # save model and apply early stopping, if applicable
        if avg_val_loss < best_val_loss:
            torch.save(model, args.model)
            best_val_loss = avg_val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement == 5:
            print('applying early stopping')
            break

# evaluate model
else:
    print('evaluation mode\n')

    # instantiate dataloader
    test_loader = data.get_data_loaders(args.data, test_loader = True, test_batch_size = args.val_batch)

    # setup progress bar
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm.tqdm(desc = desc.format(0), total = len(val_loader), leave = False, file = args.file, initial = 0)

    # evaluate model performance on test set
    model.eval()
    with torch.no_grad():
        running_test_loss = 0
        for test_batch_idx, test_data in enumerate(test_loader):
            test_inputs, _ = test_data
            test_inputs = test_inputs.to(device)

            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_inputs)

            running_test_loss += test_loss.item()
    
            # update progress bar
            pbar.desc = desc.format(test_loss.item())
            pbar.update()
    
    # print test set loss
    tqdm.tqdm.write('test loss: {:.2f}\n'.format(test_running_loss / (test_batch_idx + 1)), file = args.file)