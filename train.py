import os
import numpy as np
from config import _C as cfg
from function import ResTCN_traner
from NeuralNetwork import Network

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=1000)
    args = parser.parse_args()
    params = vars(args)
    cfg.EPOCHS = params['epochs']
    cfg.BATCHSIZE = params['batch-size']
    model = Network()
    trainer = ResTCN_trainer(model)
    all_log = []
    for epoch in range(0, params['epochs']):
        training_log = trainer.train()
        all_log.append(training_log)
