# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# define class
class Model(nn.Module):
    '''conv. autoencoder'''
    def __init__(self):
        '''constructor'''
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, padding = 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 2, stride = 2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 2, stride = 2)
        self.deconv3 = nn.ConvTranspose2d(32, 3, 2, stride = 2)
        self.conv4 = nn.Conv2d(3, 3, 5, padding = 2)

    def forward(self, x):
        '''forward prop.'''
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.conv4(x))

        return x