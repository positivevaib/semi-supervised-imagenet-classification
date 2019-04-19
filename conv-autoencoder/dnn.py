# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# define classes
class Net(nn.Module):
    '''conv. autoencoder'''
    def __init__(self):
        '''constructor'''
        super().__init__()

        self.enc1 = nn.Conv2d(3, 32, 5, padding = 2)
        self.enc2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.enc3 = nn.Conv2d(64, 128, 3, padding = 1)

        self.dec1 = nn.Conv2d(128, 64, 3, padding = 1)
        self.dec2 = nn.Conv2d(64, 32, 3, padding = 1)
        self.dec3 = nn.Conv2d(32, 3, 3, padding = 1)
        self.dec4 = nn.Conv2d(3, 3, 5, padding = 2)

    def forward(self, x):
        '''forward prop.'''
        x = F.max_pool2d(F.relu(self.enc1(x)), 2)
        x = F.max_pool2d(F.relu(self.enc2(x)), 2)
        x = F.max_pool2d(F.relu(self.enc3(x)), 2)

        x = F.interpolate(F.relu(self.dec1(x)), scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = F.interpolate(F.relu(self.dec2(x)), scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = F.interpolate(F.relu(self.dec3(x)), scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = torch.sigmoid(self.dec4(x))

        return x