import torch
from torch import optim
from config import _C as cfg
from torch.utils.tensorboard import SummaryWriter
import os


def get_optimizer(model, cfg=cfg):
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


class Logger:
    def __init__(self, log_dir, summary_writer=None):
        self.writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)
        self.step = 0

    def add_graph(self,model=None,input:tuple = None):
        self.writer.add_graph(model,input)
        self.flush()

    def log_scalar(self, scalar, name, step_):
        self.writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step,phase='train'):
        """Will log all scalars in the same plot."""
        self.writer.add_scalars(
            '{}_{}'.format(group_name,phase), scalar_dict,step)

    def log_video(self, video_frames, name, step, fps=10):
        assert len(
            video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self.writer.add_video('{}'.format(
            name), video_frames, step, fps=fps)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

def save_checkpoint(states, is_best=False, output_dir=cfg.MODEL.DIR,
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
