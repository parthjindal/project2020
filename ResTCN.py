import numpy as np
import torch
from torch import nn


class ResTCN(nn.Module):
    """
        Residual Temporal Convolution Network\n
        Input Tensor Shape: (Batch_size, features, temporal_dimension) / (N,F,T)
        Output Tensor Shape: (Batch_size, new_features, (temporal_dimension-filter_length + padding*2 +1)/stride) / (N,F',T')\n
        Layers:Batchnorm-->Relu/Activation-->1D_Convolution
    """

    def __init__(self, feature_length=None, num_filters=1, filter_length=2, stride=1, padding=0, eps=1e-5, momentum=0, **kwargs):
        super(ResTCN, self).__init__()
        self.batch_norm = nn.BatchNorm1d(
            num_features=feature_length, eps=eps, momentum=momentum)
        self.activation = nn.ReLU()
        self.reshape = nn.Conv1d(
            feature_length, num_filters, kernel_size=1, stride=stride)
        self.conv = nn.Conv1d(
            in_channels=feature_length, out_channels=num_filters, kernel_size=filter_length, stride=stride, padding=padding)

    def forward(self, x):
        orig_x = x
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv(x)
        if orig_x.shape == x.shape:
            return x+orig_x
        return x+self.reshape(orig_x)
