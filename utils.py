import torch
from torch import optim
from config import _C as cfg


def get_optimizer(model, cfg):
    optimizer = None
    if cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.SGD.LR,
            momentum=cfg.SGD.MOMENTUMS,
            weight_decay=cfg.SGD.WEIGHT_DECAY
        )

    elif cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.ADAM.LR,
            betas=cfg.ADAM.BETAS,
            weight_decay=cfg.ADAM.WEIGHT_DECAY,
            eps=cfg.ADAM.EPS
        )
    return optimizer


def create_logger():
    # TODO
    pass


def save_checkpoint(states, is_best, output_dir=cfg.MODEL.DIR,
                    filename="checkpoint.pth"):
    # output_dir = config['model_dir']
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def from_numpy(device=None, *args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
