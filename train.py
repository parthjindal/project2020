import os
import numpy as np
from config import _C as configs
from function import ResTCN_trainer
from NeuralNetwork import Network
from utils import Logger

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=1000)
    args = parser.parse_args()
    params = vars(args)

    configs.EPOCHS = params['epochs']
    configs.BATCHSIZE = params['batchsize']
    
    model = Network(cfg= configs)
    trainer = ResTCN_trainer(model)

    all_log = []
    #TODO:
    #ADD LOGGER FOR TRAINING DATA
    #APPEND LOSSES INDIVIDUALLY
    
    for epoch in range(0, params['epochs']):

        training_log = trainer.train()
        all_log.append(training_log['Loss'].sum())
        print("-"*50)
        print("Epoch: {} & Loss: {}".format(epoch,training_log["Loss"].sum()))
        print("-"*50)

if __name__ =="__main__":
    main()