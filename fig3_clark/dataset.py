import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd


class ClarkDataset(Dataset):
    def __init__(self,
                 batch_size=16,
                 num_workers=0,
                 device='cpu',
                 train_idx=None,
                 output='Butyrate'):

        self.device = torch.device(device)

        self.df = pd.read_csv('/project/vitelli/matthew/soil_microbiome_model_reduction/clark_data/2020_02_28_MasterDF_Ophelia.csv')
        self.df = self.df[self.df['Contamination?'] == 'No']

        self.targets = torch.from_numpy(self.df[output].values).float().to(self.device)
        self.inputs = torch.from_numpy(self.df.loc[:, 'BA Fraction':'B.cereus Fraction'].values).float().to(self.device)
        self.inputs[torch.isnan(self.inputs)] = 0

        self.names = self.df.loc[:, 'BA Fraction':'B.cereus Fraction'].columns
        self.names = [x.split(' ')[0] for x in self.names]

        # Remove "EH" entry (always 0)
        eh_idx = np.asarray([name == 'EH' for name in self.names])
        self.inputs = self.inputs[:, ~eh_idx]
        self.names = [name for name in self.names if name != 'EH']

        self.batch_size = batch_size

        if train_idx is None:
            self.loader_idx = np.arange(len(self))
            self.loader = self.make_loader(indices=self.loader_idx, batch_size=batch_size, num_workers=num_workers)
        elif isinstance(train_idx, list):
            self.loader_idx = train_idx
            self.loader_list = [self.make_loader(indices=idx, batch_size=batch_size, num_workers=num_workers) for idx in train_idx]
        else:
            self.loader_idx = train_idx
            self.loader = self.make_loader(indices=self.loader_idx, batch_size=batch_size, num_workers=num_workers)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {'input': self.inputs[idx], 'target': self.targets[idx], 'idx': idx}

    def make_loader(self, indices, batch_size, num_workers, pin_memory=False):
        sampler = SubsetRandomSampler(indices)
        return torch.utils.data.DataLoader(self,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory)
