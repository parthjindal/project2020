from NeuralNetwork import Network
import torch
from torch import nn
from loss import FKDLoss
import numpy as np
import utils
from utils import to_numpy
from dataset import LoadData
from config import _C as cfg
from torch.utils.data import DataLoader


class ResTCN_trainer():

    def __init__(self, model):
        self.model = model
        self.optimizer = utils.get_optimizer(model)
        self.Training_set_size = utils.get_training_set_size()
        self.batch_size = cfg.BATCHSIZE
        self.trainset = LoadData(cfg, transform=None)
        self.load_data = DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True)

    def train(self, epoch):
        # checkout https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
        '''
        # suppose you first back-propagate loss1, then loss2 (you can also do the reverse)
        loss1.backward(retain_graph=True)
        loss2.backward() # now the graph is freed, and next process of batch gradient descent is ready
        optimizer.step() # update the network parameters
        '''
        loss_history = [0., 0., 0., 0., 0.]

        for samples in self.load_data:

            x = samples['data'].to(cfg.DEVICE)
            labels = samples['label'].to(cfg.DEVICE)
            y_pred = self.model(x)
            y1, y2, y3, y4, y_hat = y_pred

            self.optimizer.zero_grad()
            loss = nn.CrossEntropyLoss(y_hat, labels, reduction='mean')
            loss.backward(retain_graph=True)
            loss_fn = FKDLoss()
            #LOSS1
            loss1 = loss_fn(y1, y_hat, labels)
            loss1.backward()
            #LOSS2
            loss2 = loss_fn(y2, y_hat, labels)
            loss2.backward()
            #LOSS3
            loss3 = loss_fn(y3, y_hat, labels)
            loss3.backward()
            #LOSS4
            loss4 = loss_fn(y4, y_hat, labels)
            loss4.backward()

            optimizer.step()
            # loss_history.append(loss1)
            # TODO: ADD LOSS_HISTORY TO LOGGER AND TENSORBOARD
            # TODO: PRETTY PRINT EPOCHS
            # TODO: CHECK IF GRADIENTS WORK???

        return{
            'Training Loss': {
                "Loss1": loss_history

            }
        }

    def validate(self):
        pass

    def cal_accuracy(self):
        pass
