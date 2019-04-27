# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# define classes
# alexnet
class AlexNet(nn.Module):
    '''feedforward neural net - alexnet'''
    def __init__(self):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 192, kernel_size = 3, padding = 2)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn45 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256*2*2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn67 = nn.BatchNorm1d(4096)
        self.out = nn.Linear(4096, 4)
        
    def forward(self, x):
        '''forward prop'''
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size = 3, stride = 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size = 3, stride = 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn45(self.conv4(x)))
        x = F.max_pool2d(F.relu(self.bn45(self.conv5(x))), kernel_size = 3, stride = 2)
        x = x.view(-1, 256*2*2)
        x = F.relu(self.bn67(self.fc1(x)))
        x = F.relu(self.bn67(self.fc2(x)))
        x = self.out(x)
        
        return x