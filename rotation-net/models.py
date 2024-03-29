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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.bn6_7 = nn.BatchNorm1d(4096)
        self.out = nn.Linear(4096, 4)

    def forward(self, x):
        '''forward prop'''
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))),
                         kernel_size=3,
                         stride=2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))),
                         kernel_size=3,
                         stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4_5(self.conv4(x)))
        x = F.max_pool2d(F.relu(self.bn4_5(self.conv5(x))),
                         kernel_size=3,
                         stride=2)
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.bn6_7(self.fc1(x)))
        x = F.relu(self.bn6_7(self.fc2(x)))
        x = self.out(x)

        return x


# resnet
class BasicIdentityBlock(nn.Module):
    '''resnet basic identity block'''

    def __init__(self, channels):
        '''constructor'''
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        '''forward prop'''
        x_orig = x
        x = F.relu(self.bn(self.conv(x)))
        x = self.bn(self.conv(x))
        x += x_orig
        x = F.relu(x)

        return x


class BasicProjectionBlock(nn.Module):
    '''resnet basic projection block'''

    def __init__(self, channels):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(channels, (channels * 2),
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.conv2 = nn.Conv2d((channels * 2), (channels * 2),
                               kernel_size=3,
                               padding=1)
        self.bn = nn.BatchNorm2d((channels * 2))
        self.proj = nn.Conv2d(channels, (channels * 2),
                              kernel_size=1,
                              stride=2)

    def forward(self, x):
        '''forward prop'''
        x_orig = x
        x_orig = self.bn(self.proj(x_orig))
        x = F.relu(self.bn(self.conv1(x)))
        x = self.bn(self.conv2(x))
        x += x_orig
        x = F.relu(x)

        return x


class BottleneckIdentityBlock(nn.Module):
    '''resnet bottleneck identity block'''

    def __init__(self, channels):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(channels, int(channels / 4), kernel_size=1)
        self.conv2 = nn.Conv2d(int(channels / 4),
                               int(channels / 4),
                               kernel_size=3,
                               padding=1)
        self.bn1_2 = nn.BatchNorm2d(int(channels / 4))
        self.conv3 = nn.Conv2d(int(channels / 4), channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        '''forward prop'''
        x_orig = x
        x = F.relu(self.bn1_2(self.conv1(x)))
        x = F.relu(self.bn1_2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += x_orig
        x = F.relu(x)

        return x


class BottleneckProjectionBlock(nn.Module):
    '''resnet bottleneck projection block'''

    def __init__(self, channels, factor=2, stride=2):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(channels,
                               int(channels / factor),
                               kernel_size=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(int(channels / factor),
                               int(channels / factor),
                               kernel_size=3,
                               padding=1)
        self.bn1_2 = nn.BatchNorm2d(int(channels / factor))
        self.conv3 = nn.Conv2d(int(channels / factor),
                               int(channels * (4 / factor)),
                               kernel_size=1)
        self.proj = nn.Conv2d(channels,
                              int(channels * (4 / factor)),
                              kernel_size=1,
                              stride=stride)
        self.bn3_proj = nn.BatchNorm2d(int(channels * (4 / factor)))

    def forward(self, x):
        '''forward prop'''
        x_orig = x
        x_orig = self.bn3_proj(self.proj(x_orig))
        x = F.relu(self.bn1_2(self.conv1(x)))
        x = F.relu(self.bn1_2(self.conv2(x)))
        x = self.bn3_proj(self.conv3(x))
        x += x_orig
        x = F.relu(x)

        return x


class ResNet18(nn.Module):
    '''feedforward neural net - resnet18'''

    def __init__(self):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(BasicIdentityBlock(64),
                                   BasicIdentityBlock(64))
        self.conv3 = nn.Sequential(BasicProjectionBlock(64),
                                   BasicIdentityBlock(128))
        self.conv4 = nn.Sequential(BasicProjectionBlock(128),
                                   BasicIdentityBlock(256))
        self.conv5 = nn.Sequential(BasicProjectionBlock(256),
                                   BasicIdentityBlock(512))
        self.out = nn.Linear(512, 4)

    def forward(self, x):
        '''forward prop'''
        x = F.max_pool2d(F.relu(self.bn(self.conv1(x))),
                         kernel_size=3,
                         stride=2,
                         padding=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.avg_pool2d(x, kernel_size=3)
        x = x.view(-1, 512)
        x = self.out(x)

        return x


class ResNet34(nn.Module):
    '''feedforward neural net - resnet34'''

    def __init__(self):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(BasicIdentityBlock(64),
                                   BasicIdentityBlock(64),
                                   BasicIdentityBlock(64))
        self.conv3 = nn.Sequential(BasicProjectionBlock(64),
                                   BasicIdentityBlock(128),
                                   BasicIdentityBlock(128),
                                   BasicIdentityBlock(128))
        self.conv4 = nn.Sequential(BasicProjectionBlock(128),
                                   BasicIdentityBlock(256),
                                   BasicIdentityBlock(256),
                                   BasicIdentityBlock(256),
                                   BasicIdentityBlock(256),
                                   BasicIdentityBlock(256))
        self.conv5 = nn.Sequential(BasicProjectionBlock(256),
                                   BasicIdentityBlock(512),
                                   BasicIdentityBlock(512))
        self.out = nn.Linear(512, 4)

    def forward(self, x):
        '''forward prop'''
        x = F.max_pool2d(F.relu(self.bn(self.conv1(x))),
                         kernel_size=3,
                         stride=2,
                         padding=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.avg_pool2d(x, kernel_size=3)
        x = x.view(-1, 512)
        x = self.out(x)

        return x


class ResNet50(nn.Module):
    '''feedforward neural net - resnet50'''

    def __init__(self):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            BottleneckProjectionBlock(64, factor=1, stride=1),
            BottleneckIdentityBlock(256), BottleneckIdentityBlock(256))
        self.conv3 = nn.Sequential(BottleneckProjectionBlock(256),
                                   BottleneckIdentityBlock(512),
                                   BottleneckIdentityBlock(512),
                                   BottleneckIdentityBlock(512))
        self.conv4 = nn.Sequential(BottleneckProjectionBlock(512),
                                   BottleneckIdentityBlock(1024),
                                   BottleneckIdentityBlock(1024),
                                   BottleneckIdentityBlock(1024),
                                   BottleneckIdentityBlock(1024),
                                   BottleneckIdentityBlock(1024))
        self.conv5 = nn.Sequential(BottleneckProjectionBlock(1024),
                                   BottleneckIdentityBlock(2048),
                                   BottleneckIdentityBlock(2048))
        self.out = nn.Linear(2048, 4)

    def forward(self, x):
        '''forward prop'''
        x = F.max_pool2d(F.relu(self.bn(self.conv1(x))),
                         kernel_size=3,
                         stride=2,
                         padding=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.avg_pool2d(x, kernel_size=3)
        x = x.view(-1, 2048)
        x = self.out(x)

        return x


class ResNet101(nn.Module):
    '''feedforward neural net - resnet101'''

    def __init__(self):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            BottleneckProjectionBlock(64, factor=1, stride=1),
            BottleneckIdentityBlock(256), BottleneckIdentityBlock(256))
        self.conv3 = nn.Sequential(BottleneckProjectionBlock(256),
                                   BottleneckIdentityBlock(512),
                                   BottleneckIdentityBlock(512),
                                   BottleneckIdentityBlock(512))
        self.conv4 = [BottleneckProjectionBlock(512)]
        for _ in range(22):
            self.conv4.append(BottleneckIdentityBlock(1024))
        self.conv4 = nn.Sequential(*self.conv4)
        self.conv5 = nn.Sequential(BottleneckProjectionBlock(1024),
                                   BottleneckIdentityBlock(2048),
                                   BottleneckIdentityBlock(2048))
        self.out = nn.Linear(2048, 4)

    def forward(self, x):
        '''forward prop'''
        x = F.max_pool2d(F.relu(self.bn(self.conv1(x))),
                         kernel_size=3,
                         stride=2,
                         padding=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.avg_pool2d(x, kernel_size=3)
        x = x.view(-1, 2048)
        x = self.out(x)

        return x


class ResNet152(nn.Module):
    '''feedforward neural net - resnet152'''

    def __init__(self):
        '''constructor'''
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(
            BottleneckProjectionBlock(64, factor=1, stride=1),
            BottleneckIdentityBlock(256), BottleneckIdentityBlock(256))
        self.conv3 = [BottleneckProjectionBlock(256)]
        for _ in range(7):
            self.conv3.append(BottleneckIdentityBlock(512))
        self.conv3 = nn.Sequential(*self.conv3)
        self.conv4 = [BottleneckProjectionBlock(512)]
        for _ in range(35):
            self.conv4.append(BottleneckIdentityBlock(1024))
        self.conv4 = nn.Sequential(*self.conv4)
        self.conv5 = nn.Sequential(BottleneckProjectionBlock(1024),
                                   BottleneckIdentityBlock(2048),
                                   BottleneckIdentityBlock(2048))
        self.out = nn.Linear(2048, 4)

    def forward(self, x):
        '''forward prop'''
        x = F.max_pool2d(F.relu(self.bn(self.conv1(x))),
                         kernel_size=3,
                         stride=2,
                         padding=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.avg_pool2d(x, kernel_size=3)
        x = x.view(-1, 2048)
        x = self.out(x)

        return x
