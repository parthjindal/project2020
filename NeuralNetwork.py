from ResTCN import ResTCN
import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class BuildBlock(nn.Module):
    def __init__(self, feature_length=None, num_filters=1, filter_length=2, stride=1, eps=1e-5, momentum=0, **kwargs):
        super(BuildBlock, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ('block1', ResTCN(feature_length, num_filters,
                              filter_length, stride, eps, momentum, **kwargs)),
            ('block2', ResTCN(num_filters, num_filters,
                              filter_length, 1, eps, momentum, **kwargs)),
            ('block3', ResTCN(num_filters, num_filters,
                              filter_length, 1, eps, momentum, **kwargs))]
        ))

    def forward(self, x):
        return self.block(x)

class Network(nn.Module):
    def __init__(self, feature_length, filter_length=8,**kwargs):
        super(Network, self).__init__()
        if 'padding' not in kwargs.keys():
            if filter_length % 2 == 0:
                padding = (int((filter_length-1)/2), int((filter_length+1)/2))
            else:
                padding = (int((filter_length-1)/2), int((filter_length-1)/2))
        else:
            padding = kwargs['padding']
        self.pad = nn.ConstantPad1d(padding,value=0)
        self.conv1 = nn.Conv1d(in_channels=feature_length,
                               out_channels=32, kernel_size=8, stride=1)
        self.Block1 = BuildBlock(32, 32, 8, 1,)
        self.Block2 = BuildBlock(32, 64, 8, 2,)
        self.Block3 = BuildBlock(64, 128, 8, 2,)
        self.Block4 = BuildBlock(128, 256, 8, 2,)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        return x


x = np.random.randn(32, 104).reshape(1, 32, 104)
x = torch.from_numpy(x).to(dtype=torch.float32).to("cuda:0")
model = Network(32).to("cuda:0")


print(model(x).shape)
