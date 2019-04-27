# import dependencies
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import data
import models
import train
import evaluate

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--architecture',
                    type=str,
                    default='alexnet',
                    help='type of model architecture to use')
parser.add_argument('--checkpoint',
                    type=str,
                    default=None,
                    help='absolute path to model checkpoint')
parser.add_argument('--data',
                    type=str,
                    default=None,
                    help='absolute path to datasets')
parser.add_argument('--environment',
                    type=str,
                    default='semi-supervised-imagenet-classification',
                    help='visdom environment name')
parser.add_argument('--evaluate',
                    action='store_true',
                    help='evaluate model performance on test set')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='initial learning rate')
parser.add_argument('--load',
                    action='store_true',
                    help='load pre-trained model')
parser.add_argument('--matplotlib',
                    action='store_true',
                    help='display matplotlib figure')
parser.add_argument('--pbar_file',
                    type=str,
                    default=None,
                    help='absolute path to file to dump stdout')
parser.add_argument('--resume',
                    action='store_true',
                    help='resume training from checkpoint')
parser.add_argument('--self_supervised',
                    action='store_true',
                    help='perform self-supervised training or evaluation')
parser.add_argument('--self_supervised_epochs',
                    type=int,
                    default=100,
                    help='total number of self-supervised training epochs')
parser.add_argument('--self_supervised_history',
                    type=str,
                    default=None,
                    help='absolute path to self-supervised training history')
parser.add_argument('--self_supervised_model',
                    type=str,
                    default=None,
                    help='absolute path to save or load self-supervised model')
parser.add_argument('--split',
                    type=float,
                    default=0.8,
                    help='training and validation split ratio')
parser.add_argument('--supervised',
                    action='store_true',
                    help='perform supervised training or evaluation')
parser.add_argument('--supervised_epochs',
                    type=int,
                    default=100,
                    help='total number of supervised training epochs')
parser.add_argument('--supervised_history',
                    type=str,
                    default=None,
                    help='absolute path to supervised training history')
parser.add_argument('--supervised_model',
                    type=str,
                    default=None,
                    help='absolute path to save or load supervised model')
parser.add_argument('--train_batch',
                    type=int,
                    default=128,
                    help='training batch size')
parser.add_argument('--val_batch',
                    type=int,
                    default=640,
                    help='validation and test batch size')
parser.add_argument('--visdom',
                    action='store_true',
                    help='create live training plots')

args = parser.parse_args()

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: {}\n'.format(device))

# instantiate model
if args.architecture == 'alexnet':
    model = models.AlexNet()
elif args.architecture == 'resnet18':
    model = models.ResNet18()
elif args.architecture == 'resnet34':
    model = models.ResNet34()
elif args.architecture == 'resnet50':
    model = models.ResNet50()
elif args.architecture == 'resnet101':
    model = models.ResNet101()
elif args.architecture == 'resnet152':
    model = models.ResNet152()

model.to(device)

# instantiate loss function
criterion = nn.CrossEntropyLoss()

# train model
if not args.evaluate:
    print('training model')

    # instantiate dataloaders
    train_loader, val_loader = data.get_data_loaders(
        args.data,
        train_ratio=args.split,
        train_batch_size=args.train_batch,
        val_batch_size=args.val_batch)

    train_size = len(train_loader) * args.train_batch
    val_size = len(val_loader) * args.val_batch

    print('train set: {} images'.format(train_size))
    print('val set: {} images\n'.format(val_size))

    # instantiate optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    # perform self-supervised training
    if args.self_supervised:
        print('self-supervised training\n')
        # load pre-trained model
        if args.load:
            print('loading pre-trained model\n')
            model.load_state_dict(
                torch.load(args.self_supervised_model, map_location=device))

        # train
        train.train(model, train_loader, train_size, val_loader, val_size,
                    criterion, optimizer, scheduler,
                    args.self_supervised_epochs, device,
                    args.self_supervised_model, args.checkpoint,
                    args.self_supervised_history, args.resume, args.visdom,
                    args.environment, args.matplotlib, args.pbar_file)

    # perform supervised training
    if args.supervised:
        print('supervised training\n')
        # prepare model for transfer learning
        for param in model.parameters():
            param.require_grad = False

        if args.architecture == 'alexnet':
            model.out = nn.Linear(4096, 1000).to(device)
        elif args.architecture == 'resnet18' or \
             args.architecture == 'resnet34':
            model.out = nn.Linear(512, 1000).to(device)
        elif args.architecture == 'resnet50' or \
             args.architecture == 'resnet101' or \
             args.architecture == 'resnet152':
            model.out = nn.Linear(2048, 1000).to(device)

        # load pre-trained model
        if args.load:
            print('loading pre-trained model\n')
            model.load_state_dict(
                torch.load(args.supervised_model, map_location=device))

        # train
        train.train(model, train_loader, train_size, val_loader, val_size,
                    criterion, optimizer, scheduler, args.supervised_epochs,
                    device, args.supervised_model, args.checkpoint,
                    args.supervised_history, args.resume, args.visdom,
                    args.environment, args.matplotlib, args.pbar_file)

# evaluate model
else:
    print('evaluating model\n')

    # load pre-trained model
    if args.load:
        print('loading pre-trained model\n')
        if args.supervised:
            if args.architecture == 'alexnet':
                model.out = nn.Linear(4096, 1000).to(device)
            elif args.architecture == 'resnet18' or \
                 args.architecture == 'resnet34':
                model.out = nn.Linear(512, 1000).to(device)
            elif args.architecture == 'resnet50' or \
                 args.architecture == 'resnet101' or \
                 args.architecture == 'resnet152':
                model.out = nn.Linear(2048, 1000).to(device)

            model.load_state_dict(
                torch.load(args.supervised_model, map_location=device))
        elif args.self_supervised:
            model.load_state_dict(
                torch.load(args.self_supervised_model, map_location=device))

    # instantiate dataloader
    dataloader = data.get_data_loaders(args.data,
                                       test_loader=True,
                                       test_batch_size=args.val_batch)
    data_size = len(dataloader) * args.val_batch

    # instantiate loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate
    evaluate.evaluate(model, dataloader, data_size, criterion, device, True,
                      args.pbar_file)
