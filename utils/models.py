import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms

act_dict = {'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'sin': torch.sin,
            'none': nn.Identity()}


class FC_predictor(nn.Module):
    def __init__(self, 
                 network_hparams={'channels': [], 'act': 'relu', 'dropout': 0., 'dropout_input': 0.},
                 optimizer_hparams={'schedule_rate': None, 'LR': None, 'type': 'adam'},
                 device='cuda:0'
                ):

        self.isnan=False
        super().__init__()
                
        
        channels = network_hparams['channels']
        act = network_hparams['act']

        self.layers = nn.ModuleList()
        self.layers.append(nn.Dropout(p=network_hparams['dropout_input']))
        for ch, ch_next in zip(channels[:-2], channels[1:-1]):
            self.layers.append(nn.Linear(ch, ch_next))
            self.layers.append(nn.Dropout(p=network_hparams['dropout']))
            self.layers.append(act_dict[act])
            
        self.layers.append(nn.Linear(channels[-2], channels[-1])) # no activation in last layer
        
        if optimizer_hparams.get('type', 'adam')=='adam':
            self.optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=optimizer_hparams['LR'])
        elif optimizer_hparams.get('type', 'adam')=='adamw':
            self.optimizer = torch.optim.AdamW([{'params': self.parameters()}], lr=optimizer_hparams['LR'],         weight_decay=optimizer_hparams.get('weight_decay', 0.01))

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, optimizer_hparams['schedule_rate']) #gamma decay rate
        
        self.device = device      
        self.to(self.device)

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
        return x
        





class FC_Gumbelpredictor(FC_predictor):
    def __init__(self, 
                 network_hparams={'inchannel': 1, 'channels': [], 'act': 'relu', 'dropout': 0., 'dropout_input': 0.},
                 optimizer_hparams={'schedule_rate': None, 'LR': None},
                 tau_hparams = {'init': 1., 'relax_rate': 0.99, 'min': 0.1},
                 device='cuda:0'
                ):

        self.early_stop=False
        self.isnan=False
        inchannel = network_hparams.pop('inchannel')
        super().__init__(network_hparams=network_hparams, 
                         optimizer_hparams=optimizer_hparams,
                         device=device)
        
        self.proj_logits = nn.Parameter(torch.zeros(inchannel, network_hparams['channels'][0]))
        self.optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=optimizer_hparams['LR'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, optimizer_hparams['schedule_rate']) #gamma decay rate
        
        self.device = device
        self.to(self.device)

        self.tau = tau_hparams['init'] # .to(self.device)
        self.tau_rate = tau_hparams['relax_rate']
        self.tau_min = tau_hparams['min']
        
    def proj_mat(self):
        return nn.functional.gumbel_softmax(self.proj_logits, tau=self.tau, hard=False, dim=1)
    
    def deterministic_proj_mat(self): # for interpretation, not used in training. This is the "P" matrix described in Methods.
        return torch.softmax(self.proj_logits, dim=1)

    def update_tau(self):
            if self.tau>self.tau_min: self.tau *= self.tau_rate
            return
    



class GatedModel(FC_Gumbelpredictor):
    def __init__(self, *args, **kwargs):
        freeze_proj = kwargs.pop('freeze_proj', False)
        super().__init__(*args, **kwargs)
 
        if freeze_proj: # only optimize non-projection parameters, so that the projection is not affected by the gate optimization
            nn_params_excluding_proj = [p for n, p in self.named_parameters() if 'proj' not in n]
            self.optimizer = torch.optim.Adam(nn_params_excluding_proj, lr=kwargs['optimizer_hparams']['LR'])
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs['optimizer_hparams']['LR'])

        self.log_gate = nn.Parameter(torch.ones(self.proj_logits.shape[0], 1)*5) # gate values will be close to one
        self.log_gate.requires_grad = True
        self.gate_optimizer = torch.optim.Adam([{'params': self.log_gate}], lr=kwargs['optimizer_hparams']['gate_LR'])

    def gate(self):
        return torch.sigmoid(self.log_gate)




