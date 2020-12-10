import argparse
import numpy as np
from config import _C as cfg
from function import ResTCN_trainer
from NeuralNetwork import Network
from utils import Logger
import torch
from torch.utils.tensorboard import SummaryWriter


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

    logger = Logger(cfg.LOGDIR)
    model = Network(cfg)
    dump_input = torch.rand(
        (1, cfg.DATASET.NUM_JOINTS, 10)
    )
    logger.add_graph(model, (dump_input,))  # Log Model Architecture
    trainer = ResTCN_trainer(model)

    for epoch in range(cfg.EPOCHS):
        training_log = trainer.train()
        print("-"*50)
        print("Epoch: {} & Loss: {}".format(epoch, training_log["Loss"]))
        print("-"*50)
        logger.log_scalars(training_log, "Training",logger.step)
        logger.step += 1


if __name__ == "__main__":
    main()
