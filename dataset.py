import torch
from torch.utils.data import Dataset, DataLoader
from config import _C
import pandas as pd
import numpy as np


class LoadData(Dataset):
    def __init__(self, cfg, transform=None):
        super(LoadData, self).__init__()
        self.data = pd.read_csv(cfg.DATASET.TRAIN_SET+'.csv')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = np.asarray(self.data.iloc[index, 1:], dtype=np.float)
        label = self.data.iloc[index, 0]
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
