from ResTCN import ResTCN
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from utils import *
from yacs import _C as cfg

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
    def __init__(self, feature_length=None, cfg=None, **kwargs):
        super(Network, self).__init__()

        f = cfg.FILTER_LENGTH
        if 'padding' not in kwargs.keys():  # Initialize Padding
            if f % 2 == 0:
                padding = (int((f-1)/2), int((f+1)/2))
            else:
                padding = (int((f-1)/2), int((f-1)/2))
        else:
            padding = kwargs['padding']

        # Manual Padding Layer For 'even' Filter length
        self.pad = nn.ConstantPad1d(padding, value=0)
        self.conv1 = nn.Conv1d(in_channels=feature_length,
                               out_channels=cfg.BLOCK1.NUM_FILTERS, kernel_size=cfg.FILTER_LENGTH, stride=1)

        self.Block1 = BuildBlock(
            cfg.BLOCK1.NUM_FILTERS, cfg.BLOCK1.NUM_FILTERS, cfg.FILTER_LENGTH, cfg.BLOCK1.STRIDE)
        self.Block2 = BuildBlock(
            cfg.BLOCK1.NUM_FILTERS, cfg.BLOCK2.NUM_FILTERS, cfg.FILTER_LENGTH, cfg.BLOCK2.STRIDE)
        self.Block3 = BuildBlock(
            cfg.BLOCK2.NUM_FILTERS, cfg.BLOCK3.NUM_FILTERS, cfg.FILTER_LENGTH, cfg.BLOCK3.STRIDE)
        self.Block4 = BuildBlock(
            cfg.BLOCK3.NUM_FILTERS, cfg.BLOCK4.NUM_FILTERS, cfg.FILTER_LENGTH, cfg.BLOCK4.STRIDE)

    def forward(self, x):
        ########TODO: ADD FCs,GLOBAL_AVG_POOLINGs,CONCATENATION,CHANGE RETURN TO (PRED1,PRED2,PRED3,X)#####
        x = self.pad(x)
        x = self.conv1(x)
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        return x
