import torch
from torch import Tensor
from torch.nn import functional as F


class FKDLoss():
    '''
    Fusion Knowledge Distillation\n
    params: input: subblock.logit shape:(N,C)\n
    params: soft_target: Fusion.logit shape:(N,C)\n
    params: labels: one-hot-encoded.label shape:(N,)\n
    output KL_Divergence over input,soft_target - CrossEntropy(input,labels)
    '''

    def __init__(self, temp=1) -> None:
        self.T = temp

    def forward(self, input: Tensor, soft_target: Tensor, labels: Tensor) -> Tensor:
        # TODO: Check if labels shape is 2 or 1 ??????
        labels = labels.reshape((labels.shape[0],))
        return F.kl_div(F.log_softmax(input/self.T, dim=1), F.softmax(soft_target/self.T, dim=1), reduction='batchmean') + F.cross_entropy(input, labels, reduction='mean')
    __call__ = forward
