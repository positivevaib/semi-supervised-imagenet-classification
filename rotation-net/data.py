# import dependencies
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# define function
def get_data_loaders(path,
                     test_loader=False,
                     train_ratio=0.8,
                     train_batch_size=1,
                     val_batch_size=1,
                     test_batch_size=1):
    '''load dataset'''
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                             inplace=True)
    ])
    dataset = datasets.ImageFolder(path, transform=transformations)

    if not test_loader:
        train_set_size = int(train_ratio * len(dataset))
        val_set_size = len(dataset) - train_set_size
        train_data, val_data = data.random_split(
            dataset, (train_set_size, val_set_size))

        train_loader = data.DataLoader(train_data,
                                       batch_size=train_batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(val_data,
                                     batch_size=val_batch_size,
                                     shuffle=False)

        return train_loader, train_set_size, val_loader, val_set_size
    else:
        test_loader = data.DataLoader(dataset,
                                      batch_size=test_batch_size,
                                      shuffle=False)

        return test_loader, len(dataset)
