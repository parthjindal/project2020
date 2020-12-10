import argparse
import numpy as np
import os
from config import _C as cfg
from function import ResTCN_trainer
from NeuralNetwork import Network
from utils import *
import torch
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str)

    parser.add_argument('--batchsize',
                        help='batchsize',
                        type=int, required=False)

    parser.add_argument('--epochs',
                        help='episodes',
                        type=int, required=False)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.cfg is not None:
        cfg.merge_from_file(args.cfg)
    if args.batchsize is not None:
        cfg.EPOCHS = params['epochs']
    if args.epochs is not None:
        cfg.BATCHSIZE = params['batchsize']
    cfg.merge_from_list(args.opts)


def main():

    parse_args()
    seed = torch.seed()%20
    log_dir = cfg.TENSORBOARD_DIR + cfg.MODEL.NAME + "/"+str(seed)
    logger = Logger(log_dir)
    model_dir = cfg.MODEL.DIR + cfg.MODEL.NAME + "/"+str(seed)
    if os.path.isdir(model_dir)!=True:
        os.makedirs(model_dir)

    model = Network(cfg)
    dump_input = torch.rand(
        (1, cfg.DATASET.NUM_JOINTS,cfg.DEFAULT_FRAMES)
    )
    logger.add_graph(model, (dump_input,))  # Log Model Architecture
    
    #Toggle to get model summary
    summary(model,dump_input.shape[1:],batch_size=32,device="cpu")
    trainer = ResTCN_trainer(model)

    optimizer = trainer.optimizer
    print("------STARTING TRAINING-------")
    
    for epoch in range(cfg.EPOCHS):

        training_log = trainer.train()
        print("-"*50)
        print("Epoch: {} & Loss: {}".format(epoch, training_log["Loss"]))
        print("-"*50)
        logger.log_scalars(training_log,logger.step)
        logger.step += 1

        if epoch % cfg.SAVE_FREQUENCY == 0:
            perf_indicator = trainer.cal_accuracy()
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME+str(seed),
                'state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, output_dir= model_dir)


if __name__ == "__main__":
    main()
