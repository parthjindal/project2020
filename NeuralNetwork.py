from ResTCN import ResTCN
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from configs import config


class BuildBlock(nn.Module):
    def __init__(self, feature_length=None, num_filters=1, filter_length=1, stride=(1, 1, 1), eps=1e-5, momentum=0, **kwargs):
        super(BuildBlock, self).__init__()

        if type(stride) is int:  # Stride for All 3 Subblocks
            stride = (stride,)+2*(1,)
        elif type(stride) is tuple:
            stride = stride + (3-len(stride))*(1,)

        self.block = nn.Sequential(OrderedDict([
            ('blockA', ResTCN(feature_length, num_filters,
                              filter_length, stride[0], eps, momentum, **kwargs)),
            ('blockB', ResTCN(num_filters, num_filters,
                              filter_length, stride[1], eps, momentum, **kwargs)),
            ('blockC', ResTCN(num_filters, num_filters,
                              filter_length, stride[2], eps, momentum, **kwargs))]
        ))

    def forward(self, x):
        return self.block(x)


class Network(nn.Module):
    def __init__(self, feature_length, filter_length=8, **kwargs):
        super(Network, self).__init__()

        if 'padding' not in kwargs.keys():  # Initialize Padding
            if filter_length % 2 == 0:
                padding = (int((filter_length-1)/2), int((filter_length+1)/2))
            else:
                padding = (int((filter_length-1)/2), int((filter_length-1)/2))
        else:
            padding = kwargs['padding']

        self.pad = nn.ConstantPad1d(padding, value=0)  # Manual Padding Layer For 'even' Filter length
        self.conv1 = nn.Conv1d(in_channels=feature_length,
                               out_channels=32, kernel_size=config['filter_length'], stride=1)
        self.Block1 = BuildBlock(32, 32, config['filter_length'], 1)
        self.Block2 = BuildBlock(32, 64, config['filter_length'], 2)
        self.Block3 = BuildBlock(64, 128, config['filter_length'], 2)
        self.Block4 = BuildBlock(128, 256, config['filter_length'], 2)

    def forward(self, x):
        ########TODO: ADD FCs,GLOBAL_AVG_POOLINGs,CONCATENATION,CHANGE RETURN TO (PRED1,PRED2,PRED3,X)#####
        x = self.pad(x)
        x = self.conv1(x)
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        return x
