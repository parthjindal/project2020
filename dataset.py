import torch
from torch.utils.data import Dataset, DataLoader
from config import _C 
import pandas as pd
import numpy as np
from sklearn import preprocessing
from config import _C as cfg


class LoadData(Dataset):
    def __init__(self, type, transform=None):
        super(LoadData, self).__init__()
        self.data = pd.read_csv(type+'.csv')
        self.transform = transform
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.data.iloc[:,0])

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, index):
        #TODO: ENCODE LABELS FROM 0-(cfg.CLASSES-1)
        data = np.asarray(self.data.iloc[index, 1:], dtype=np.float).reshape((cfg.DATASET.NUM_JOINTS,-1))
        label = self.label_encoder.transform([self.data.iloc[index, 0]])
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        #sample['label'] = self.onehotencoder(sample['label'])    
        return sample
