from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 4

_C.OPTIMIZER = "adam"

_C.ADAM =CN()
_C.ADAM = CN()
_C.ADAM.LR = 1E-5
_C.ADAM.EPS = 1E-8
_C.ADAM.BETAS = (0.9,0.999)
_C.ADAM.WEIGHT_DECAY = 0

_C.SGD = CN()
_C.SGD.LR = 1E-4
_C.SGD.MOMENTUM = 0
_C.SGD.WEIGHT_DECAY = 0

_C.EPOCHS = 1000
_C.BATCHSIZE = 256
_C.LOGDIR = ""
_C.TENSORBOARD_DIR = "./tb/"
_C.MODEL = CN()
_C.MODEL.DIR = "./models/"

_C.GPUS = (0,)
_C.FILTER_LENGTH = 8
_C.FPS = 3
_C.INPUT_DIR = ""
_C.TEMPERATURE = 1

_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'csv'
_C.DATASET.NUM_JOINTS = 17

_C.BLOCK1 = CN()
_C.BLOCK1.NUM_FILTERS = 32
_C.BLOCK1.STRIDE = (1,1,1)

_C.BLOCK2 = CN()
_C.BLOCK2.NUM_FILTERS = 64
_C.BLOCK2.STRIDE = (2,1,1)

_C.BLOCK3 = CN()
_C.BLOCK3.NUM_FILTERS = 128
_C.BLOCK3.STRIDE = (2,1,1)

_C.BLOCK4 = CN()
_C.BLOCK4.NUM_FILTERS = 256
_C.BLOCK4.STRIDE = (2,1,1)

# def update_config(cfg, args):
#     cfg.defrost()
#     cfg.merge_from_file(args.cfg)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
