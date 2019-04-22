import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import numpy as np
def MnistLabel(class_num):
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    class_tot = [0] * 10
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 10 * class_num:
                break
    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))

def MnistUnlabel():
    raw_dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    return raw_dataset
def MnistTest():
    return datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       #transforms.Normalize((0.1307,), (0.3081,))
                   ]))
def ImageNetSupervised():
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)])
    dataset = datasets.ImageFolder("/supervised", transform = transformations)
    return dataset
    """
    class_tot = [0] * 1000
    data = []
    labels = []
    positive_tot = 0
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    for i in range(raw_dataset.__len__()):
        datum, label = raw_dataset.__getitem__(perm[i])
        if class_tot[label] < class_num:
            data.append(datum.numpy())
            labels.append(label)
            class_tot[label] += 1
            tot += 1
            if tot >= 10 * class_num:
                break
    return TensorDataset(torch.FloatTensor(np.array(data)), torch.LongTensor(np.array(labels)))
    """
def ImageNetUnsupervised():
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)])
    dataset = datasets.ImageFolder("/unsupervised", transform = transformations)
    return dataset

def ImageNetTest():
    # Alfredo
