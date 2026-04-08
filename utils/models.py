import numpy as np
import torch
import torch.nn as nn

act_dict = {'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'none': nn.Identity()}


class FC_predictor(nn.Module):
    def __init__(self,
                 network_hparams={'channels': [], 'act': 'relu', 'dropout': 0., 'dropout_input': 0.},
                 optimizer_hparams={'schedule_rate': None, 'LR': None},
                 device='cuda:0'):

        self.isnan = False
        super().__init__()

        channels = network_hparams['channels']
        act = network_hparams['act']

        self.layers = nn.ModuleList()
        self.layers.append(nn.Dropout(p=network_hparams['dropout_input']))
        for ch, ch_next in zip(channels[:-2], channels[1:-1]):
            self.layers.append(nn.Linear(ch, ch_next))
            self.layers.append(nn.Dropout(p=network_hparams['dropout']))
            self.layers.append(act_dict[act])

        self.layers.append(nn.Linear(channels[-2], channels[-1]))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=optimizer_hparams['LR'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, optimizer_hparams['schedule_rate'])

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
                 tau_hparams={'init': 1., 'relax_rate': 0.99, 'min': 0.1},
                 device='cuda:0'):

        self.early_stop = False
        self.isnan = False

        # Don't mutate caller's dict — build parent hparams without 'inchannel'
        inchannel = network_hparams['inchannel']
        parent_hparams = {k: v for k, v in network_hparams.items() if k != 'inchannel'}

        super().__init__(network_hparams=parent_hparams,
                         optimizer_hparams=optimizer_hparams,
                         device=device)

        self.proj_logits = nn.Parameter(torch.zeros(inchannel, network_hparams['channels'][0]))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=optimizer_hparams['LR'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, optimizer_hparams['schedule_rate'])

        self.device = device
        self.to(self.device)

        self.tau = tau_hparams['init']
        self.tau_rate = tau_hparams['relax_rate']
        self.tau_min = tau_hparams['min']

    def proj_mat(self):
        return nn.functional.gumbel_softmax(self.proj_logits, tau=self.tau, hard=False, dim=1)

    def deterministic_proj_mat(self):
        return torch.softmax(self.proj_logits, dim=1)

    def update_tau(self):
        if self.tau > self.tau_min:
            self.tau *= self.tau_rate


class GatedModel(FC_Gumbelpredictor):
    def __init__(self, *args, **kwargs):
        # Don't mutate caller's kwargs
        freeze_proj = kwargs.get('freeze_proj', False)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'freeze_proj'}

        super().__init__(*args, **filtered_kwargs)

        if freeze_proj:
            nn_params_excluding_proj = [p for n, p in self.named_parameters() if 'proj' not in n]
            self.optimizer = torch.optim.Adam(nn_params_excluding_proj, lr=kwargs['optimizer_hparams']['LR'])
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs['optimizer_hparams']['LR'])

        self.log_gate = nn.Parameter(torch.ones(self.proj_logits.shape[0], 1) * 5)
        self.log_gate.requires_grad = True
        self.gate_optimizer = torch.optim.Adam([{'params': self.log_gate}], lr=kwargs['optimizer_hparams']['gate_LR'])

    def gate(self):
        return torch.sigmoid(self.log_gate)


def post_training_evaluate(checkpoint_path, DatasetClass, ModelClass, is_gated=False, taxa_key=None, device='cpu'):
    """Load a checkpoint, reconstruct model + datasets, run forward pass, return augmented results dict.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        DatasetClass: Dataset class to reconstruct train/test data.
        ModelClass: Model class (FC_Gumbelpredictor or GatedModel).
        is_gated: Whether model uses a gate (GatedModel).
        taxa_key: If not None, sample['input'] is a dict and this key selects the taxa.
        device: Device string.
    """
    x = torch.load(checkpoint_path, map_location=device)

    x['train_dataset_kwargs']['device'] = device
    x['test_dataset_kwargs']['device'] = device
    dataset_train = DatasetClass(**x['train_dataset_kwargs'])
    dataset_test = DatasetClass(**x['test_dataset_kwargs'])

    # Determine n_pcs and ensemble index from filename
    fname = checkpoint_path.split('/')[-1]
    npcs = int(fname.split('Npcs-')[1].split('_')[0])
    ensidx = int(fname.split('model-')[1].split('.')[0])

    # model_kwargs in checkpoint may be {npcs: kwargs} or {taxa: {npcs: kwargs}}
    mk = x['model_kwargs']
    if npcs in mk:
        model_kwargs = mk[npcs]
    else:
        model_kwargs = mk
    model_kwargs['device'] = device
    if is_gated:
        model_kwargs['freeze_proj'] = False
    model = ModelClass(**model_kwargs)
    model.load_state_dict(x['model_state_dicts'])
    model.tau = x['curr_tau'] / 10
    model.to(device)
    model.eval()

    sample_train = next(iter(dataset_train.loader_list[ensidx]))
    sample_test = next(iter(dataset_test.loader_list[ensidx]))

    def _get_input(sample):
        return sample['input'][taxa_key] if taxa_key else sample['input']

    with torch.no_grad():
        proj = model.proj_mat().cpu().numpy()
        proj_det = model.deterministic_proj_mat().cpu().numpy()

        if is_gated:
            gate = model.gate().cpu().numpy()
            train_inpt = (_get_input(sample_train) * model.gate().T) @ model.proj_mat()
            test_inpt = (_get_input(sample_test) * model.gate().T) @ model.proj_mat()
        else:
            gate = None
            train_inpt = _get_input(sample_train) @ model.proj_mat()
            test_inpt = _get_input(sample_test) @ model.proj_mat()

        pred_train = model(train_inpt, None).cpu().numpy()
        pred_test = model(test_inpt, None).cpu().numpy()
        target_train = sample_train['target'].cpu().numpy()
        target_test = sample_test['target'].cpu().numpy()

    r2_train = 1 - np.mean((pred_train - target_train) ** 2) / np.var(target_train)
    r2_test = 1 - np.mean((pred_test - target_test) ** 2) / np.var(target_test)

    x['r2_train'] = r2_train
    x['r2_test'] = r2_test
    x['pred_train'] = pred_train
    x['pred_test'] = pred_test
    x['target_train'] = target_train
    x['target_test'] = target_test
    x['proj'] = proj
    x['proj_det'] = proj_det
    x['gate'] = gate
    x['logits'] = x['proj_logits'].detach().cpu().numpy()

    return x




