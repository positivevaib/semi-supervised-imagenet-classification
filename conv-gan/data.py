import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_labeled_train_data_tensor(path, samples_class):
    transformations = transforms.Compose([
        transforms.Resize(28, interpolation=2),
        transforms.ToTensor()#,
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True)
        ])
    raw_dataset = datasets.ImageFolder(os.path.join(path, "supervised/train"), transform = transformations)

    # we create a tensor dataset where we limit the number of samples per class
    class_tot = [0] * 1000
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(len(raw_dataset))
    for i in range(len(raw_dataset)):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < samples_class:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 1000 * samples_class:
                break
    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))

def get_unlabeled_data(path):
    transformations = transforms.Compose([
        transforms.Resize(28, interpolation=2),
        transforms.ToTensor()#
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])
    
    dataset = datasets.ImageFolder(os.path.join(path, "unsupervised"), transform = transformations)
    return dataset

def get_labeled_test_data_loader(path, batch_size = 1):
    '''load dataset'''
    transformations = transforms.Compose([
        transforms.Resize(28, interpolation=2),
        transforms.ToTensor()#,
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True)
        ])
    dataset = datasets.ImageFolder(os.path.join(path, "supervised/val"), transform = transformations)
    loader = data.DataLoader(dataset, batch_size = batch_size, shuffle = False)
    return loader