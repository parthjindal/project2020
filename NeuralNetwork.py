from ResTCN import ResTCN
import torch as tf
from torch import nn
import numpy as np
from collections import OrderedDict
from utils import *
from config import _C as cfg

class BuildBlock(nn.Module):
    def __init__(self, feature_length=None, num_filters=1, filter_length=1, stride=(1, 1, 1), eps=1e-5, momentum=0, **kwargs):
        super(BuildBlock, self).__init__()

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
    def __init__(self, feature_length=17, cfg=cfg, **kwargs):
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
        self.linear1 = nn.Linear(in_features=cfg.BLOCK1.NUM_FILTERS,out_features=cfg.OUTPUT_CLASSES)
        self.linear2 = nn.Linear(in_features=cfg.BLOCK2.NUM_FILTERS,out_features=cfg.OUTPUT_CLASSES)
        self.linear3 = nn.Linear(in_features=cfg.BLOCK3.NUM_FILTERS,out_features=cfg.OUTPUT_CLASSES)
        self.linear4 = nn.Linear(in_features=cfg.BLOCK4.NUM_FILTERS,out_features=cfg.OUTPUT_CLASSES)
        self.linear5 = nn.Linear(in_features=4*cfg.OUTPUT_CLASSES,out_features=cfg.OUTPUT_CLASSES)
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
        x1 = self.Block1(x)
        x = x + x1
        x1 = x.mean(dim=(2))
        x1 = self.linear1(x1)
        x2 = self.Block2(x)
        x = x + x2
        x2 = x.mean(dim=(2))
        x2 = self.linear2(x2)
        x3 = self.Block3(x)
        x = x + x3
        x3 = x.mean(dim=(2))
        x3 = self.linear3(x3)
        x4 = self.Block4(x)
        x = x + x4
        x4 = x.mean(dim=(2))
        x4 = self.linear3(x4)
        x = tf.cat(x1,x2,x3,x4)
        x = self.linear5(x)
        return x1,x2,x3,x4,x
