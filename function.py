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
        self.batch_size = cfg.BATCHSIZE
        self.trainset = LoadData(cfg, transform=None)
        self.load_data = DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True)
        #self.Training_set_size = len(self.trainset)

    def train(self):
        # checkout https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
        '''
        # suppose you first back-propagate loss1, then loss2 (you can also do the reverse)
        loss1.backward(retain_graph=True)
        loss2.backward() # now the graph is freed, and next process of batch gradient descent is ready
        optimizer.step() # update the network parameters
        '''
        loss_history = [0., 0., 0., 0., 0.]

        for samples in self.load_data:
            print(samples)
            x = torch.DoubleTensor(samples['data']).to(cfg.DEVICE)
            labels = torch.DoubleTensor(samples['label']).to(cfg.DEVICE)
            y_pred = self.model(x)
            y1, y2, y3, y4, y_hat = y_pred

            self.optimizer.zero_grad()
            loss = nn.CrossEntropyLoss(y_hat, labels, reduction='mean')
            loss.backward(retain_graph=True)
            loss_fn = FKDLoss()
            loss_history[0]=utils.to_numpy(loss)
            #LOSS1
            loss1 = loss_fn(y1, y_hat, labels)
            loss1.backward()
            loss_history[1]=utils.to_numpy(loss1)
            #LOSS2
            loss2 = loss_fn(y2, y_hat, labels)
            loss2.backward()
            loss_history[2]=utils.to_numpy(loss2)
            #LOSS3
            loss3 = loss_fn(y3, y_hat, labels)
            loss3.backward()
            loss_history[3]=utils.to_numpy(loss3)
            #LOSS4
            loss4 = loss_fn(y4, y_hat, labels)
            loss4.backward()
            loss_history[4]=utils.to_numpy(loss4)

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
