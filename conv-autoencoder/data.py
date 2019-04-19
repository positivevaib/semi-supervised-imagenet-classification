# import dependencies
import torch
import torchvision

# define function
def load_dataset(path, training_batch_size = 1):
    '''load dataset'''
    transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), 
                                                                                    (0.229, 0.224, 0.225),
                                                                                    inplace=True)])
    #transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace = True)])
    dataset = torchvision.datasets.ImageFolder(path, transform = transformations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = training_batch_size, shuffle = True)
    
    return dataloader