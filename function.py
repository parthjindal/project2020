from NeuralNetwork import Network
import torch
from torch import nn
from loss import FKDLoss
import numpy as np
import utils
from utils import to_numpy, Logger
from dataset import LoadData
from config import _C as cfg
from torch.utils.data import DataLoader


class ResTCN_trainer():

    def __init__(self, model):
        self.model = model.to(device=cfg.DEVICE)
        self.optimizer = utils.get_optimizer(model)
        self.batch_size = cfg.BATCHSIZE
        self.trainset = LoadData(cfg, transform=None)
        self.load_data = DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True)
        self.loss_fn = FKDLoss(cfg.TEMPERATURE)
        self.logger = Logger(cfg.LOGDIR)

    def train(self):
        loss_history = np.ndarray((5,))

        for samples in self.load_data:
            x = (samples['data']).to(cfg.DEVICE, dtype=torch.float32).reshape(
                (samples['data'].shape[0], cfg.DATASET.NUM_JOINTS, -1))
            labels = (samples['label']).to(
                cfg.DEVICE, dtype=torch.long).reshape((samples['label'].shape[0],))
            y1, y2, y3, y4, y_hat = self.model(x)
            self.optimizer.zero_grad()
            loss = nn.CrossEntropyLoss(reduction='mean')(y_hat, labels)

            loss.backward(retain_graph=True)
            loss_history[0] += utils.to_numpy(loss)

            loss1 = self.loss_fn(y1, y_hat, labels)
            loss1.backward(retain_graph=True)
            loss_history[1] += utils.to_numpy(loss1)

            # LOSS2
            for name, parameter in self.model.named_parameters():
                if 'conv1' in name or 'linear1' in name or 'Block1' in name:
                    parameter.requires_grad = False

            loss2 = self.loss_fn(y2, y_hat, labels)
            loss2.backward(retain_graph=True)
            loss_history[2] += utils.to_numpy(loss2)

            # LOSS3
            for name, parameter in self.model.named_parameters():
                if 'conv2' in name or 'linear2' in name or 'Block2' in name:
                    parameter.requires_grad = False

            loss3 = self.loss_fn(y3, y_hat, labels)
            loss3.backward(retain_graph=True)
            loss_history[3] += utils.to_numpy(loss3)

            # LOSS4
            for name, parameter in self.model.named_parameters():
                if 'conv3' in name or 'linear3' in name or 'Block3' in name:
                    parameter.requires_grad = False

            loss4 = self.loss_fn(y4, y_hat, labels)
            loss4.backward(retain_graph=True)
            loss_history[4] += utils.to_numpy(loss4)

            for parameter in self.model.parameters():  # set requires_grad to True for all layers
                parameter.requires_grad = True

            self.optimizer.step()  # take optimizer step after all grads accumulated

        return{
            "Loss": loss_history[0],
            "Loss1": loss_history[1],
            "Loss2": loss_history[2],
            "Loss3": loss_history[3],
            "Loss4": loss_history[4]
        }

    def validate(self):
        pass

    def cal_accuracy(self):
        pass
