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
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transformations)

    # we create a tensor dataset where we limit the number of samples per class
    class_tot = [0] * 10
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
            if tot >= 10 * samples_class:
                break
    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))

def get_unlabeled_data(path):
    transformations = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transformations)
    return raw_dataset

def get_labeled_test_data_loader(path, batch_size = 1):
    transformations = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
    
    raw_dataset = datasets.MNIST('../data', train=False, download=True,
                   transform=transformations)
    return raw_dataset