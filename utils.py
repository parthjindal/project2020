import torch
from torch import optim
from configs import config


def get_optimizer(model):
    optimizer = None
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['sgd']['lr'],
            momentum=config['sgd']['momentum'],
            weight_decay=config['sgd']['weight_decay'],
        )
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['adam']['lr'],
            betas=config['adam']['betas'],
            weight_decay=config['adam']['weight_decay'],
            eps=config['adam']['eps']
        )
    return optimizer

def create_logger():
    ###TODO
    pass

def save_checkpoint(states, is_best, output_dir=None,
                    filename="checkpoint.pth"):
    output_dir = config['model_dir']
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))
