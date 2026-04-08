import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd


class SoilDatasetFiltered(Dataset):
    def __init__(self,
                 batch_size=16,
                 num_workers=0,
                 CHL=0,
                 device='cpu',
                 sample_replicates_as_one=False,
                 comp_data_rescale={'type': 'log', 'EPS': 1e-4, 'add': 10},
                 input_type='composition',
                 pred_vars='',
                 train_idx=None,
                 sort=True,
                 normalize_target=False):

        self.normalize_target = normalize_target
        self.taxa_ordered_names = np.load(
            '/project/vitelli/matthew/soil_microbiome_model_reduction/filtered_projection_names_ASV_to_taxa.npy',
            allow_pickle=True).item()

        VARIABLE_MAP = {'no3': 1, 'no2': 0, 'nh4': 2}
        self.sort = sort
        self.input_type = input_type
        self.pred_var_idx = [VARIABLE_MAP[x] for x in pred_vars.split(',')]
        self.device = torch.device(device)

        filestr = 'dataframe_composition_and_time_series_filtered_enrichment_fixedTime_CHL'

        df_chl_neg = pd.read_pickle(f'/project/vitelli/matthew/soil_microbiome_model_reduction/data_processed/{filestr}-{0}.df')
        df_chl_pos = pd.read_pickle(f'/project/vitelli/matthew/soil_microbiome_model_reduction/data_processed/{filestr}-{1}.df')

        # Remove bad soils (present in chl_neg but not chl_pos)
        bad_soils = [{'Unit': 12, 'Soil': 'Soil15'},
                     {'Unit': 40, 'Soil': 'Soil12'},
                     {'Unit': 20, 'Soil': 'Soil12'}]
        for bad_soil in bad_soils:
            bad_idx = df_chl_neg[(df_chl_neg.Unit == bad_soil['Unit']) & (df_chl_neg.Soil == bad_soil['Soil'])].index
            df_chl_neg = df_chl_neg.drop(index=bad_idx)
        df_chl_neg.reset_index(inplace=True, drop=True)

        # Reset to unique condition indices
        m = np.arange(df_chl_neg.Condition_idx.max() + 1)
        m[df_chl_neg.Condition_idx.unique()] = np.arange(len(df_chl_neg.Condition_idx.unique()))
        df_chl_neg['Condition_idx'] = m[df_chl_neg['Condition_idx']]

        # Copy enrichment values from CHL neg to CHL pos
        keys = ['logratio_asv_unfilt', 'logratio_family_unfilt', 'logratio_phylum_unfilt',
                'logratio_asv_filt', 'logratio_family_filt', 'logratio_phylum_filt']
        df_chl_pos.loc[:, keys] = df_chl_neg.loc[:, keys]

        # Combine dataframes
        df_chl_pos['Condition_idx'] += df_chl_neg['Condition_idx'].max() + 1
        if CHL == 'pos':
            df = df_chl_pos
        elif CHL == 'both':
            df = pd.concat([df_chl_neg, df_chl_pos], axis=0).reset_index()
        else:
            df = df_chl_neg

        self.df_chl_pos = df_chl_pos.rename(columns={'index': 'Replicate'})
        self.df = df.rename(columns={'index': 'Replicate'})

        self.soil_ph = np.asarray([x for x in self.df.Soil_pH.values])
        self.ph = np.asarray([x for x in self.df.pH.values])
        self.no2 = np.asarray([x for x in self.df.NO2.values])
        self.no3 = np.asarray([x for x in self.df.NO3.values])
        self.nh4 = np.asarray([x for x in self.df.NH4.values])

        self.comp_t0_asv = np.asarray([x for x in self.df.comp_T0_asv_filt.values])
        self.comp_t0_family = np.asarray([x for x in self.df.comp_T0_family_filt.values])
        self.comp_t0_phylum = np.asarray([x for x in self.df.comp_T0_phylum_filt.values])
        self.comp_tT_asv = np.asarray([x for x in self.df.comp_T9_asv_filt.values])
        self.comp_tT_family = np.asarray([x for x in self.df.comp_T9_family_filt.values])
        self.comp_tT_phylum = np.asarray([x for x in self.df.comp_T9_phylum_filt.values])

        self.enrich_asv = np.asarray([x for x in self.df.logratio_asv_filt.values])
        self.enrich_family = np.asarray([x for x in self.df.logratio_family_filt.values])
        self.enrich_phylum = np.asarray([x for x in self.df.logratio_phylum_filt.values])

        # Sort by mean abundance
        if self.sort:
            self.sort_idx = {
                'Family': np.argsort(np.mean(self.comp_tT_family, axis=0))[::-1],
                'Phylum': np.argsort(np.mean(self.comp_tT_phylum, axis=0))[::-1],
                'ASV': np.argsort(np.mean(self.comp_tT_asv, axis=0))[::-1],
            }
            self.comp_t0_asv = self.comp_t0_asv[:, self.sort_idx['ASV']]
            self.comp_t0_family = self.comp_t0_family[:, self.sort_idx['Family']]
            self.comp_t0_phylum = self.comp_t0_phylum[:, self.sort_idx['Phylum']]
            self.comp_tT_asv = self.comp_tT_asv[:, self.sort_idx['ASV']]
            self.comp_tT_family = self.comp_tT_family[:, self.sort_idx['Family']]
            self.comp_tT_phylum = self.comp_tT_phylum[:, self.sort_idx['Phylum']]
            self.enrich_asv = self.enrich_asv[:, self.sort_idx['ASV']]
            self.enrich_family = self.enrich_family[:, self.sort_idx['Family']]
            self.enrich_phylum = self.enrich_phylum[:, self.sort_idx['Phylum']]

        self.comp_taxa_tT = {'ASV': self.comp_tT_asv, 'Family': self.comp_tT_family, 'Phylum': self.comp_tT_phylum}
        self.comp_taxa_t0 = {'ASV': self.comp_t0_asv, 'Family': self.comp_t0_family, 'Phylum': self.comp_t0_phylum}
        self.log_enrich_taxa = {'ASV': self.enrich_asv, 'Family': self.enrich_family, 'Phylum': self.enrich_phylum}

        # Load to device
        self.soil_ph = torch.from_numpy(self.soil_ph).float().to(self.device)
        self.ph = torch.from_numpy(self.ph).float().to(self.device)
        self.no2 = torch.from_numpy(self.no2).float().to(self.device)
        self.no3 = torch.from_numpy(self.no3).float().to(self.device)
        self.nh4 = torch.from_numpy(self.nh4).float().to(self.device)

        self.no2_normalized = self.no2 / self.no2.std()
        self.no3_normalized = self.no3 / self.no3.std()
        self.nh4_normalized = self.nh4 / self.nh4.std()

        self.comp_taxa_t0 = {tx: torch.from_numpy(self.comp_taxa_t0[tx]).float().to(self.device) for tx in self.comp_taxa_t0}
        self.comp_taxa_tT = {tx: torch.from_numpy(self.comp_taxa_tT[tx]).float().to(self.device) for tx in self.comp_taxa_tT}
        self.log_enrich_taxa = {tx: torch.from_numpy(self.log_enrich_taxa[tx]).float().to(self.device) for tx in self.log_enrich_taxa}

        self.inputs = self.log_enrich_taxa if self.input_type == 'enrichment' else self.comp_taxa_tT
        self.batch_size = batch_size

        assert len(self) % 3 == 0, f"len is not a multiple of 3. Len = {len(self)}"

        if train_idx is None:
            if sample_replicates_as_one:
                condition_idx = np.arange(len(self.df.Condition_idx.unique()))
                self.loader_idx = self.df.index[self.df.Condition_idx.isin(condition_idx)].values
            else:
                self.loader_idx = np.arange(len(self))
            self.loader = self.make_loader(indices=self.loader_idx, batch_size=batch_size, num_workers=num_workers)
        elif isinstance(train_idx, list):
            if sample_replicates_as_one:
                self.loader_idx = [self.df.index[self.df.Condition_idx.isin(idx)].values for idx in train_idx]
            else:
                self.loader_idx = train_idx
            self.loader_list = [self.make_loader(indices=idx, batch_size=batch_size, num_workers=num_workers) for idx in self.loader_idx]
        else:
            if sample_replicates_as_one:
                self.loader_idx = self.df.index[self.df.Condition_idx.isin(train_idx)].values
            else:
                self.loader_idx = train_idx
            self.loader = self.make_loader(indices=self.loader_idx, batch_size=batch_size, num_workers=num_workers)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        no2 = self.no2[idx]
        no3 = self.no3[idx]
        nh4 = self.nh4[idx]

        no2_norm = self.no2_normalized[idx]
        no3_norm = self.no3_normalized[idx]
        nh4_norm = self.nh4_normalized[idx]

        dyn = torch.stack([no2, no3, nh4])
        dyn_norm = torch.stack([no2_norm, no3_norm, nh4_norm])

        input = {tx: self.inputs[tx][idx] for tx in self.inputs}
        target = dyn[self.pred_var_idx, :].view(-1)
        target_norm = dyn_norm[self.pred_var_idx, :].view(-1)

        return {'input': input,
                'target': target if not self.normalize_target else target_norm,
                'target_norm': target_norm,
                'no2': no2, 'no3': no3, 'nh4': nh4,
                'dyn': dyn,
                'pH': self.ph[idx].unsqueeze(0),
                'soil_pH': self.soil_ph[idx].unsqueeze(0),
                'idx': idx}

    def make_loader(self, indices, batch_size, num_workers, pin_memory=False):
        sampler = SubsetRandomSampler(indices)
        return torch.utils.data.DataLoader(self,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory)
