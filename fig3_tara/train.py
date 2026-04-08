import os
import numpy as np
import pandas as pd
import torch
import time
import argparse
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import FC_Gumbelpredictor
from dataset import TaraDataset


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
    ap.add_argument("--targets", type=str)
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
    ap.add_argument("--patience", type=int, default=1000, help='Early stopping patience (steps without test loss improvement)')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    source_file_path = '/project/vitelli/matthew/soil_microbiome_model_reduction/tara_data/all_environmental_variables_with_abundances.csv'
    datadf = pd.read_csv(source_file_path)
    n_samples = datadf.loc[:, args.targets.split(',')].notna().sum(axis=0).min()
    print("Number of samples with all targets:", n_samples)

    rng = np.random.default_rng(args.seed)
    train_idx = [rng.choice(n_samples, size=int(n_samples * 0.8), replace=False) for _ in range(args.N_ensemble)]
    test_idx = [np.asarray([x for x in range(n_samples) if x not in idx]) for idx in train_idx]

    train_dataset_kwargs = {
        'batch_size': len(train_idx[0]),
        'num_workers': 0,
        'device': 'cuda:0',
        'train_idx': train_idx,
        'source_file_path': source_file_path,
        'targets': args.targets.split(','),
    }
    test_dataset_kwargs = {
        'batch_size': len(test_idx[0]),
        'num_workers': 0,
        'device': 'cuda:0',
        'train_idx': test_idx,
        'source_file_path': source_file_path,
        'targets': args.targets.split(','),
    }

    dataset_train = TaraDataset(**train_dataset_kwargs)
    dataset_test = TaraDataset(**test_dataset_kwargs)
    print("Dataset shape", dataset_train.df.shape)

    # Build models
    ensemble_size = args.N_ensemble
    x = dataset_train[0]
    channel_y = len(args.targets.split(','))

    model_kwargs_all = {}
    models = {}
    channel_x_in = x['input'].shape[0]
    for n_clust in [1, 2, 3, 4, 6, 8]:
        models[n_clust] = []
        model_kwargs = {
            'network_hparams': {
                'inchannel': channel_x_in,
                'channels': (n_clust, *([args.nchannels] * args.n_layers), channel_y),
                'act': args.act,
                'dropout': args.dropout,
                'dropout_input': args.dropout,
            },
            'optimizer_hparams': {'schedule_rate': args.schedule_rate, 'LR': args.lr},
            'tau_hparams': {'init': args.tau_init, 'relax_rate': args.tau_relax_rate, 'min': args.tau_min},
        }
        model_kwargs_all[n_clust] = model_kwargs
        for idx in range(ensemble_size):
            model = FC_Gumbelpredictor(**model_kwargs)
            model.to(device)
            model.device = device
            model.train()
            models[n_clust].append(model)

    # Training loop
    count = 0
    t0 = time.time()
    train_losses = {nc: [[] for _ in range(len(models[nc]))] for nc in models}
    test_losses = {nc: [[] for _ in range(len(models[nc]))] for nc in models}

    # Early stopping state: best test loss and steps since last improvement
    best_test_loss = {nc: [float('inf')] * len(models[nc]) for nc in models}
    steps_without_improvement = {nc: [0] * len(models[nc]) for nc in models}
    early_stopped = {nc: [False] * len(models[nc]) for nc in models}

    while count < args.n_steps:
        for ens_idx in range(ensemble_size):
            nsamples = 0
            for sample, sample_test in zip(dataset_train.loader_list[ens_idx], dataset_test.loader_list[ens_idx]):
                nsamples += 1
                if ens_idx == 0:
                    count += 1
                for m, nc_idx in enumerate(models):
                    model = models[nc_idx][ens_idx]
                    if model.isnan or early_stopped[nc_idx][ens_idx]:
                        continue

                    model.optimizer.zero_grad()
                    proj_inpt = sample['input'] @ model.proj_mat()
                    pred = model(proj_inpt, None)
                    assert pred.shape == sample['target'].shape, "Prediction shape does not match target shape"

                    loss = torch.nn.functional.mse_loss(pred.squeeze(), sample['target'].squeeze())

                    with torch.no_grad():
                        test_proj_inpt = sample_test['input'] @ model.proj_mat()
                        test_pred = model(test_proj_inpt, None).squeeze()
                        test_loss = torch.nn.functional.mse_loss(test_pred, sample_test['target'].squeeze())

                    loss.backward()
                    model.optimizer.step()
                    model.update_tau()

                    if np.isnan(loss.detach().cpu().numpy()):
                        model.isnan = True
                        train_losses[nc_idx][ens_idx].append(np.nan)
                        test_losses[nc_idx][ens_idx].append(np.nan)

                    # Early stopping check
                    tl = test_loss.item()
                    if tl < best_test_loss[nc_idx][ens_idx]:
                        best_test_loss[nc_idx][ens_idx] = tl
                        steps_without_improvement[nc_idx][ens_idx] = 0
                    else:
                        steps_without_improvement[nc_idx][ens_idx] += 1
                    if steps_without_improvement[nc_idx][ens_idx] >= args.patience:
                        early_stopped[nc_idx][ens_idx] = True
                        print(f"Early stopping: nc={nc_idx}, ens={ens_idx} at step {count}", flush=True)

                    if count % args.save_freq == 0:
                        train_losses[nc_idx][ens_idx].append(loss.detach().cpu().numpy())
                        test_losses[nc_idx][ens_idx].append(test_loss.detach().cpu().numpy())

                        torch.save({
                            'model_state_dicts': model.state_dict(),
                            'proj_logits': model.proj_logits.detach().cpu().numpy(),
                            'proj': model.proj_mat().cpu().detach().numpy(),
                            'proj_det': model.deterministic_proj_mat().cpu().detach().numpy(),
                            'pred_train': pred.detach().cpu().numpy(),
                            'target_train': sample['target'].detach().cpu().numpy(),
                            'pred_test': test_pred.detach().cpu().numpy(),
                            'target_test': sample_test['target'].detach().cpu().numpy(),
                            'r2_train': compute_r2(sample['target'].detach().cpu().numpy().squeeze(), pred.detach().cpu().numpy().squeeze()),
                            'r2_test': compute_r2(sample_test['target'].detach().cpu().numpy().squeeze(), test_pred.detach().cpu().numpy().squeeze()),
                            'curr_tau': model.tau,
                            'tau_hparams': model_kwargs_all[nc_idx]['tau_hparams'],
                            'ens_idx': ens_idx,
                            'seed': args.seed,
                            'train_idx_split': np.asarray(train_idx[ens_idx]),
                            'test_idx_split': np.asarray(test_idx[ens_idx]),
                            'train_losses': train_losses[nc_idx][ens_idx],
                            'test_losses': test_losses[nc_idx][ens_idx],
                            'train_dataset_kwargs': train_dataset_kwargs,
                            'test_dataset_kwargs': test_dataset_kwargs,
                            'model_kwargs': model_kwargs_all,
                        }, os.path.join(args.save_dir, f'Npcs-{nc_idx}_model-{ens_idx}.pt'))

            assert nsamples == 1, "Batch size not equal to dataset size! Saving frequency will be incorrect"

        # Stop entirely if all models are done (nan or early stopped)
        all_done = all(
            model.isnan or early_stopped[nc_idx][ens_idx]
            for nc_idx in models
            for ens_idx, model in enumerate(models[nc_idx])
        )
        if all_done:
            print("All models early-stopped or NaN. Ending training at step %d." % count, flush=True)
            break

        if count % args.save_freq == 0:
            print(f"Step: {count} Time: {time.time() - t0:.3f} Train: {loss.item():.3f} Test: {test_loss.item():.3f}", flush=True)
