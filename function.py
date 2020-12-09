from NeuralNetwork import Network
import torch
from loss import FKDLoss
import numpy as np
import utils
from config import _C as cfg

class ResTCN_trainer():

    def __init__(self,model):
        self.optimizer = utils.get_optimizer(model)
        self.Training_set_size = utils.get_training_set_size()
        self.batch_size = cfg.BATCHSIZE
        #self.x
        #self.y
    def train(self):
        #checkout https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
        '''
        # suppose you first back-propagate loss1, then loss2 (you can also do the reverse)
        loss1.backward(retain_graph=True)
        loss2.backward() # now the graph is freed, and next process of batch gradient descent is ready
        optimizer.step() # update the network parameters
        '''
        for i in range(0,self.Training_set_size/self.batch_size+1):
            x,y = get_batch(self.x,self.y,batch_size)
            y_pred = model.forward(x)
            y1,y2,y3,y4,y5 = y_pred
            lossfunction = FKDLoss()
            optimizer.zero_grad()
            loss1 = lossfunction(y5,y4,y)
            loss2 = lossfunction(y5,y3,y)
            loss3 = lossfunction(y5,y2,y)
            loss4 = lossfunction(y5,y1,y)
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss3.backward(retain_graph=True)
            loss4.backward()
            optimizer.step()
        return{
            'Training Loss': utils.to_numpy(loss1)+utils.to_numpy(loss2)+utils.to_numpy(loss3)+utils.to_numpy(loss4)
        }

    def validate(self):
        pass

    def cal_accuracy(self):
        pass
