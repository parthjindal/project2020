import os
import numpy as np
from config import _C as cfg
from function import ResTCN_trainer
from NeuralNetwork import Network

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=1000)
    args = parser.parse_args()
    params = vars(args)
    cfg.EPOCHS = params['epochs']
    cfg.BATCHSIZE = params['batchsize']
    model = Network()
    trainer = ResTCN_trainer(model)
    all_log = []
    for epoch in range(0, params['epochs']):
        training_log = trainer.train()
        all_log.append(training_log)
        print("Epoch: {} & Loss: {}".format(epoch,training_log[0]))

if __name__ =="__main__":
    main()
