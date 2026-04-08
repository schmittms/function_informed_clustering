import os
import numpy as np
import torch
import time
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import FC_Gumbelpredictor
from dataset import SoilDatasetFiltered


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, required=True)
    ap.add_argument("--save_freq", type=int, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--CHL", type=str, default='0')
    ap.add_argument("--taxa", type=str, default='')
    ap.add_argument("--N_clusters", type=int)
    ap.add_argument("--N_ensemble", type=int)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tau_init", type=float, default=1.)
    ap.add_argument("--tau_relax_rate", type=float, default=0.99)
    ap.add_argument("--tau_min", type=float, default=0.1)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--act", type=str, default='relu')
    ap.add_argument("--nchannels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--schedule_rate", type=float, default=1.0)
    ap.add_argument("--opt_method", type=str, default='adam')
    ap.add_argument("--early_stopping_limit", type=int, default=10)
    args = ap.parse_args()

    TAXA = args.taxa.split(',')
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    nsample = 127
    rng = np.random.default_rng(args.seed)
    train_idx = [rng.choice(nsample, size=int(nsample * 0.8), replace=False) for _ in range(args.N_ensemble)]
    test_idx = [np.asarray([x for x in range(nsample) if x not in idx]) for idx in train_idx]

    train_dataset_kwargs = {
        'batch_size': len(train_idx[0]) * 3,
        'device': device,
        'num_workers': 0,
        'CHL': args.CHL,
        'sample_replicates_as_one': True,
        'input_type': 'composition',
        'pred_vars': 'no3',
        'train_idx': train_idx,
    }
    test_dataset_kwargs = {
        'batch_size': len(test_idx[0]) * 3,
        'device': device,
        'num_workers': 0,
        'CHL': args.CHL,
        'sample_replicates_as_one': True,
        'input_type': 'composition',
        'pred_vars': 'no3',
        'train_idx': test_idx,
    }

    dataset_train = SoilDatasetFiltered(**train_dataset_kwargs)
    dataset_test = SoilDatasetFiltered(**test_dataset_kwargs)
    print("Dataset shape", dataset_train.df.shape)

    # Build models
    ensemble_size = args.N_ensemble
    x = dataset_train[0]
    channel_y = x['target'].shape[0]

    model_kwargs_all = {}
    models = {}
    for t in TAXA:
        models[t] = {}
        model_kwargs_all[t] = {}
        channel_x_in = x['input'][t].shape[0]
        for n_pcs in range(1, args.N_clusters + 1):
            models[t][n_pcs] = []
            model_kwargs = {
                'network_hparams': {
                    'inchannel': channel_x_in,
                    'channels': (n_pcs, *([args.nchannels] * args.n_layers), channel_y),
                    'act': args.act,
                    'dropout': args.dropout,
                    'dropout_input': 0.0,
                },
                'optimizer_hparams': {'schedule_rate': args.schedule_rate, 'LR': args.lr},
                'tau_hparams': {'init': args.tau_init, 'relax_rate': args.tau_relax_rate, 'min': args.tau_min},
            }
            model_kwargs_all[t][n_pcs] = model_kwargs
            for idx in range(ensemble_size):
                model = FC_Gumbelpredictor(**model_kwargs)
                model.to(device)
                model.device = device
                model.train()
                models[t][n_pcs].append(model)

    # Training loop
    count = 0
    t0 = time.time()
    train_losses = {t: {pc: [[] for _ in range(len(models[t][pc]))] for pc in models[t]} for t in models}
    test_losses = {t: {pc: [[] for _ in range(len(models[t][pc]))] for pc in models[t]} for t in models}

    while count < args.n_steps:
        for ens_idx in range(ensemble_size):
            nsamples = 0
            for sample, sample_test in zip(dataset_train.loader_list[ens_idx], dataset_test.loader_list[ens_idx]):
                nsamples += 1
                if ens_idx == 0:
                    count += 1
                for t, tx in enumerate(TAXA):
                    for m, pc_idx in enumerate(models[tx]):
                        model = models[tx][pc_idx][ens_idx]
                        if model.isnan:
                            continue

                        model.optimizer.zero_grad()
                        inpt = sample['input'][tx] @ model.proj_mat()
                        pred = model(inpt, None)
                        loss = torch.nn.functional.mse_loss(pred, sample['target'])

                        with torch.no_grad():
                            test_inpt = sample_test['input'][tx] @ model.proj_mat()
                            test_pred = model(test_inpt, None)
                            test_loss = torch.nn.functional.mse_loss(test_pred, sample_test['target'])

                        loss.backward()
                        model.optimizer.step()
                        model.update_tau()

                        if np.isnan(loss.detach().cpu().numpy()):
                            model.isnan = True

                        if count % args.save_freq == 0:
                            train_losses[tx][pc_idx][ens_idx].append(loss.detach().cpu().numpy())
                            test_losses[tx][pc_idx][ens_idx].append(test_loss.detach().cpu().numpy())

                            if len(train_losses[tx][pc_idx][ens_idx]) > args.early_stopping_limit:
                                if np.min(test_losses[tx][pc_idx][ens_idx][:-args.early_stopping_limit]) < np.min(test_losses[tx][pc_idx][ens_idx][-args.early_stopping_limit + 1:]):
                                    model.early_stop = True
                                    print(f"Early stopping model {pc_idx} ensemble {ens_idx} at step {count}")

                            torch.save({
                                'model_state_dicts': model.state_dict(),
                                'args': vars(args),
                                'proj': model.proj_mat().cpu().detach().numpy(),
                                'proj_det': model.deterministic_proj_mat().cpu().detach().numpy(),
                                'proj_logits': model.proj_logits.detach().cpu().numpy(),
                                'pred_train': pred.detach().cpu().numpy(),
                                'target_train': sample['target'].detach().cpu().numpy(),
                                'pred_test': test_pred.detach().cpu().numpy(),
                                'target_test': sample_test['target'].detach().cpu().numpy(),
                                'r2_train': compute_r2(sample['target'].detach().cpu().numpy(), pred.detach().cpu().numpy()),
                                'r2_test': compute_r2(sample_test['target'].detach().cpu().numpy(), test_pred.detach().cpu().numpy()),
                                'curr_tau': model.tau,
                                'tau_hparams': model_kwargs_all[tx][pc_idx]['tau_hparams'],
                                'ens_idx': ens_idx,
                                'taxon': tx,
                                'seed': args.seed,
                                'train_idx_split': np.asarray(train_idx[ens_idx]),
                                'test_idx_split': np.asarray(test_idx[ens_idx]),
                                'train_losses': train_losses[tx][pc_idx][ens_idx],
                                'test_losses': test_losses[tx][pc_idx][ens_idx],
                                'train_dataset_kwargs': train_dataset_kwargs,
                                'test_dataset_kwargs': test_dataset_kwargs,
                                'model_kwargs': model_kwargs_all[tx],
                            }, os.path.join(args.save_dir, f'tax-{tx}_Npcs-{pc_idx}_model-{ens_idx}.pt'))

            assert nsamples == 1, "Batch size not equal to dataset size! Saving frequency will be incorrect"

        if count % args.save_freq == 0:
            n_pc = next(iter(models[TAXA[0]].keys()))
            output_str = f'\tLoss:\t{train_losses[TAXA[0]][n_pc][-1][-1]:0.4f}' + \
                         f'\tTest:\t{test_losses[TAXA[0]][n_pc][-1][-1]:0.4f}'
            print("Step: %d Time: %.3f" % (count, time.time() - t0), flush=True)
            print(output_str, flush=True)
