import torch
from torch.utils.data import Dataset, DataLoader
from config import _C 
import pandas as pd
import numpy as np
from sklearn import preprocessing


class LoadData(Dataset):
    def __init__(self, cfg, transform=None):
        super(LoadData, self).__init__()
        self.data = pd.read_csv(cfg.DATASET.TRAIN_SET+'.csv')
        self.transform = transform
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.data.iloc[:,0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #TODO: ENCODE LABELS FROM 0-(cfg.CLASSES-1)
        data = np.asarray(self.data.iloc[index, 1:], dtype=np.float).reshape((17,-1))
        label = self.label_encoder.transform([self.data.iloc[index, 0]])
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
