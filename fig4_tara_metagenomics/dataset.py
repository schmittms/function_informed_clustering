import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd


class TaraDatasetMetagenomicModules(Dataset):
    def __init__(self,
                 batch_size=16,
                 num_workers=0,
                 device='cpu',
                 targets=None,
                 train_idx=None,
                 condition_by=None):

        self.device = torch.device(device)
        self.condition_by = condition_by

        self.df = pd.read_csv('/project/vitelli/matthew/soil_microbiome_model_reduction/tara_data/metagenomic_and_environmental_variables.csv')

        if self.condition_by is not None:
            self.df = self.df[(self.df[self.condition_by['key']] <= self.condition_by['max']) &
                              (self.df[self.condition_by['key']] >= self.condition_by['min'])].reset_index(drop=True)

        abd_cols = self.df.columns.str.startswith('M')
        inputs_w_nans = torch.from_numpy(self.df.loc[:, abd_cols].values).float().to(self.device)
        targets_w_nans = torch.from_numpy(self.df.loc[:, targets].values).float().to(self.device)

        nan_samples = torch.isnan(targets_w_nans).any(dim=1)
        self.nan_samples = nan_samples
        self.inputs = inputs_w_nans[~nan_samples, :]
        self.targets = targets_w_nans[~nan_samples, :]
        self.targets /= self.targets.std(dim=0)

        self.module_names = self.df.columns[abd_cols]
        self.inputs = self.inputs / 100_000

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
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input': self.inputs[idx], 'target': self.targets[idx], 'idx': idx}

    def make_loader(self, indices, batch_size, num_workers, pin_memory=False):
        sampler = SubsetRandomSampler(indices)
        return torch.utils.data.DataLoader(self,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory)
