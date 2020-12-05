from ResTCN import ResTCN
import torch
from torch import nn
import numpy as np
from collections import OrderedDict


class BuildBlock(nn.Module):
    def __init__(self, feature_length=None, num_filters=1, filter_length=2, stride=1, padding=0, eps=1e-5, momentum=0, **kwargs):
        super(BuildBlock, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ('block1', ResTCN(feature_length, num_filters,
                              filter_length, stride, padding, eps, momentum, **kwargs)),
            ('block2', ResTCN(num_filters, num_filters,
                              filter_length, stride=1, padding, eps, momentum, **kwargs)),
            ('block3', ResTCN(num_filters, num_filters,
                              filter_length, stride=1, padding, eps, momentum, **kwargs))]
        ))

    def forward(self, x):
        return self.block(x)