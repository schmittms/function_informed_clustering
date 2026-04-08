import os
import time
import argparse
from itertools import product

import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution, root_scalar
from scipy.integrate import solve_ivp

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import GatedModel
from dataset import SoilDatasetFiltered


def parse_float_list(spec: str):
    spec = spec.strip()
    if not spec:
        return []
    return [float(x) for x in spec.split(',') if x.strip()]


def parse_subroots(spec: str):
    spec = spec.strip()
    if not spec:
        return []
    return [x.strip() for x in spec.split(';') if x.strip()]


def default_subroots():
    return [
        'gatedgumbel_250930_094936_targets-no3_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_12',
        'gatedgumbel_250930_100500_targets-no3,no2_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_13',
        'gatedgumbel_250930_102344_targets-no2_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_14',
        'gatedgumbel_250930_133017_targets-no3,nh4_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_15',
        'gatedgumbel_250930_133607_targets-nh4_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_16',
        'gatedgumbel_250930_134549_targets-no3,no2,nh4_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_17',
        'gatedgumbel_250930_135104_targets-no3_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_18',
        'gatedgumbel_250930_141401_targets-no3,no2_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_19',
        'gatedgumbel_250930_141530_targets-no2_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_20',
        'gatedgumbel_250930_184740_targets-no3,nh4_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_21',
        'gatedgumbel_250930_195055_targets-nh4_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_22',
        'gatedgumbel_250930_200004_targets-no3,no2,nh4_N-128_L-2_DO-0.15_LR-1e-2_gateLR-1e-3_beta-1e-1_TAU-2._0.9999_0.1_ACT-gelu_NENS-12_Nsteps-20000_23',
    ]


def calculate_error(params,
                    x0,
                    growth_data, growth_time, growth_xT,
                    chl_data, chl_time,
                    gamma=np.ones(2),
                    mass_weight=1e-2,
                    r_C=1.0,
                    return_full_trajectories=False,
                    log_biomass=False):
    rA1, rA2, t_switch = params
    a0_chl = chl_data[0]
    a0_growth = growth_data[0]

    t_growth, a_growth_obs = growth_time, growth_data
    t_chl, a_chl_obs = chl_time, chl_data

    term_exponential_mass0 = (x0[0] / gamma[0]) * (np.exp(gamma[0] * np.minimum(t_growth, t_switch)) - 1)
    term_exponential_mass1 = (x0[1] / gamma[1]) * (np.exp(gamma[1] * np.minimum(t_growth, t_switch)) - 1)
    term_exponential = rA1 * term_exponential_mass0 + rA2 * term_exponential_mass1

    term_linear_mass0 = x0[0] * np.exp(gamma[0] * t_switch) * np.maximum(t_growth - t_switch, 0)
    term_linear_mass1 = x0[1] * np.exp(gamma[1] * t_switch) * np.maximum(t_growth - t_switch, 0)
    term_linear = rA1 * term_linear_mass0 + rA2 * term_linear_mass1

    A_growth_model = np.maximum(a0_growth - term_exponential - term_linear, 0)
    A_chl_model = np.maximum(a0_chl - np.sum(x0[:, None] * np.asarray([rA1, rA2])[:, None] * t_chl[None, :], axis=0), 0)

    x_final_model = x0 * np.exp(gamma * t_switch)

    mse_growth = np.mean((a_growth_obs - A_growth_model) ** 2)
    mse_chl = np.mean((a_chl_obs - A_chl_model) ** 2)

    if log_biomass:
        mse_biomass = (np.log10(x_final_model + 1e-5) - np.log10(growth_xT + 1e-5)) ** 2
    else:
        mse_biomass = (x_final_model - growth_xT) ** 2

    total_error_nitrate = (mse_growth + mse_chl) / (len(t_growth) + len(t_chl))
    total_error = total_error_nitrate + np.mean(mse_biomass) * mass_weight

    if not return_full_trajectories:
        return total_error

    x_traj = x0[:, None] * np.exp(gamma[:, None] * np.minimum(t_growth, t_switch)[None, :])
    sol = root_scalar(
        lambda C0: C0 - (x0[0] / gamma[0]) * (np.exp(gamma[0] * t_switch) - 1) - r_C * (x0[1] / gamma[1]) * (np.exp(gamma[1] * t_switch) - 1),
        bracket=[0, 100],
        method='brentq',
    )
    C = sol.root - term_exponential_mass0 - r_C * term_exponential_mass1
    C = np.maximum(C, 0)

    x_pred_growth = x_traj
    x_pred_chl = x0[:, None] * np.ones_like(t_chl[None, :])

    return A_growth_model, A_chl_model, x_pred_growth, x_pred_chl, C, mse_growth, mse_chl, mse_biomass, total_error


def integrate_eom(params, a0_chl, a0_growth, x0, time_growth, time_chl, gamma, C0, r_C, K_C=1e-3, K_A=1e-3):
    rA1, rA2, _ = params

    def eom_growth(t, y):
        A, C, x1, x2 = y
        dA_dt = - (rA1 * x1 + rA2 * x2) * A / (K_A + A)
        dC_dt = - (x1 + r_C * x2) * C / (K_C + C)
        dx1_dt = gamma[0] * x1 * A / (K_A + A) * C / (K_C + C)
        dx2_dt = gamma[1] * x2 * A / (K_A + A) * C / (K_C + C)
        return [dA_dt, dC_dt, dx1_dt, dx2_dt]

    def eom_chl(t, y):
        A, C, x1, x2 = y
        dA_dt = - (rA1 * x1 + rA2 * x2) * A / (K_A + A)
        dC_dt = - (x1 + r_C * x2) * C / (K_C + C)
        dx1_dt = 0
        dx2_dt = 0
        return [dA_dt, dC_dt, dx1_dt, dx2_dt]

    y0_growth = [a0_growth, C0, x0[0], x0[1]]
    sol_growth = solve_ivp(eom_growth, [time_growth[0], time_growth[-1]], y0_growth, t_eval=time_growth, atol=1e-9, rtol=1e-9)

    y0_chl = [a0_chl, C0, x0[0], x0[1]]
    sol_chl = solve_ivp(eom_chl, [time_chl[0], time_chl[-1]], y0_chl, t_eval=time_chl, atol=1e-9, rtol=1e-9)

    return np.maximum(sol_growth.y[0], 0), sol_growth.y[1], sol_growth.y[2:4], np.maximum(sol_chl.y[0], 0), sol_chl.y[1], sol_chl.y[2:4]


def load_loss_df(masterroot, subroots, output_name='no3', tax_name='ASV', device='cpu'):
    rows = []
    t_load = time.time()

    candidate_files = []
    for sr in subroots:
        root = os.path.join(masterroot, sr)
        if not os.path.isdir(root):
            continue
        files = os.listdir(root)
        for file in files:
            if file.endswith('.pt'):
                candidate_files.append((sr, root, file))

    total_candidates = len(candidate_files)
    print(f"[load] start: {total_candidates} candidate .pt files", flush=True)

    for file_idx, (sr, root, file) in enumerate(candidate_files, start=1):
        if (file_idx == 1) or (file_idx % 10 == 0) or (file_idx == total_candidates):
            elapsed_load = time.time() - t_load
            eta_load = (elapsed_load / file_idx) * (total_candidates - file_idx)
            print(f"[load] idx={file_idx}/{total_candidates}, elapsed={elapsed_load:.1f}s, eta={eta_load:.1f}s", flush=True)

        try:
            npcs = int(file.split('_')[1].split('-')[-1])
            tax = file.split('_')[0].split('-')[-1]
            ensidx = int(file.split('_')[2].split('-')[-1].split('.')[0])
        except Exception:
            continue

        if npcs != 2 or tax != tax_name:
            continue

        path = os.path.join(root, file)
        x = torch.load(path, map_location=device)

        if x['model_kwargs'][1]['optimizer_hparams']['LR'] < 1e-3:
            continue
        if x['model_kwargs'][1]['network_hparams']['dropout'] != 0.15:
            continue
        if x['train_dataset_kwargs']['CHL'] != 'both':
            continue
        if x['train_dataset_kwargs']['pred_vars'] != output_name:
            continue

        x['train_dataset_kwargs']['device'] = device
        x['test_dataset_kwargs']['device'] = device
        dataset_train = SoilDatasetFiltered(**x['train_dataset_kwargs'])
        dataset_test = SoilDatasetFiltered(**x['test_dataset_kwargs'])

        model_kwargs = x['model_kwargs'][npcs]
        model_kwargs['network_hparams']['dropout_input'] = 0.0
        model_kwargs['device'] = device

        model = GatedModel(**model_kwargs, freeze_proj=False)
        model.load_state_dict(x['model_state_dicts'])
        model.tau = x['curr_tau'] / 10
        model.to(device)
        model.eval()

        sample_train = next(iter(dataset_train.loader_list[ensidx]))
        sample_test = next(iter(dataset_test.loader_list[ensidx]))

        inpt = (sample_train['input'][tax] * model.gate().T) @ model.proj_mat()
        pred = model(inpt, None).detach().cpu().numpy().squeeze()
        targ = sample_train['target'].detach().cpu().numpy().squeeze()

        test_inpt = (sample_test['input'][tax] * model.gate().T) @ model.proj_mat()
        test_pred = model(test_inpt, None).detach().cpu().numpy().squeeze()
        test_targ = sample_test['target'].detach().cpu().numpy().squeeze()

        r2_test = 1 - np.mean((test_pred - test_targ) ** 2) / np.var(test_targ)

        rows.append({
            'outputs': x['train_dataset_kwargs']['pred_vars'],
            'tax': tax,
            'npcs': npcs,
            'ensidx': ensidx,
            'r2_test': r2_test,
            'proj': model.proj_mat().cpu().detach().numpy(),
            'proj_det': model.deterministic_proj_mat().cpu().detach().numpy(),
            'gate': model.gate().cpu().detach().numpy(),
            'train_dataset_kwargs': x['train_dataset_kwargs'],
            'test_dataset_kwargs': x['test_dataset_kwargs'],
        })

    elapsed_load_total = time.time() - t_load
    print(f"[load] completed: {len(rows)} accepted rows, elapsed={elapsed_load_total:.1f}s", flush=True)

    return pd.DataFrame(rows)


def prepare_group_data(loss_df, n_keep=5, output_name='no3', ensemble_rank=1, device='cpu'):
    mask = loss_df.outputs == output_name

    ensembles = loss_df[mask].ensidx.values
    r2 = loss_df[mask].r2_test.values
    r2_sorted_idx = np.argsort(r2)[::-1]
    ensembles_to_keep = ensembles[r2_sorted_idx[:n_keep]]

    train_kwargs = loss_df.loc[mask].iloc[0]['train_dataset_kwargs'].copy()
    train_kwargs['device'] = device
    dataset_train = SoilDatasetFiltered(**train_kwargs)

    abd_init = dataset_train.comp_t0_asv
    abd_final = dataset_train.comp_tT_asv

    group_abd = {}
    for ens_idx in ensembles_to_keep:
        row = loss_df.loc[mask & (loss_df.ensidx == ens_idx)].iloc[0]
        proj = row['proj_det']
        gate = row['gate']

        gp_abd_init = abd_init @ (proj * gate)
        gp_abd_final = abd_final @ (proj * gate)

        idx_sort = np.argsort(np.mean(gp_abd_final, axis=0))
        group_abd[ens_idx] = {
            'init': gp_abd_init[:, idx_sort],
            'final': gp_abd_final[:, idx_sort],
        }

    chosen_ens = ensembles_to_keep[ensemble_rank]
    x_init = group_abd[chosen_ens]['init']
    x_final = group_abd[chosen_ens]['final']

    CHL_mask = dataset_train.df.Chloramphenicol == 1

    no3_growth = dataset_train.no3[~CHL_mask].numpy()
    no3_chl = dataset_train.no3[CHL_mask].numpy()
    time_growth = np.asarray([np.asarray(x) for x in dataset_train.df.Time[~CHL_mask].values]) / 24
    time_chl = np.asarray([np.asarray(x) for x in dataset_train.df.Time[CHL_mask].values]) / 24

    x_init_growth = x_init[~CHL_mask]
    x_final_growth = x_final[~CHL_mask]

    return {
        'dataset_train': dataset_train,
        'CHL_mask': CHL_mask,
        'x_init_growth': x_init_growth,
        'x_final_growth': x_final_growth,
        'no3_growth': no3_growth,
        'no3_chl': no3_chl,
        'time_growth': time_growth,
        'time_chl': time_chl,
        'ensembles_to_keep': ensembles_to_keep,
        'chosen_ens': int(chosen_ens),
    }


def run_grid_search(data, gamma0_grid, gamma1_grid, rC_grid, mass_weight_grid,
                    sample_subset_size, de_maxiter, de_popsize, de_tol, seed=0):
    rng = np.random.default_rng(seed)

    no3_growth = data['no3_growth']
    no3_chl = data['no3_chl']
    x_init_growth = data['x_init_growth']
    x_final_growth = data['x_final_growth']
    time_growth = data['time_growth']
    time_chl = data['time_chl']

    bounds = [(0.0, 5.0), (0.0, 5.0), (0.0, 4.0)]

    sample_subset = rng.choice(np.arange(len(no3_growth)), size=sample_subset_size, replace=False)
    param_grid = list(product(gamma0_grid, gamma1_grid, rC_grid, mass_weight_grid))

    rows = []
    t0 = time.time()
    print(f'[grid] total parameter sets: {len(param_grid)}', flush=True)
    print(f'[grid] subset size: {len(sample_subset)}', flush=True)

    for p, params in enumerate(param_grid):
        set_start = time.time()
        gamma = np.asarray([params[0], params[1]])
        r_C = params[2]
        mass_weight = params[3]
        print(
            f"[grid] parameter set {p + 1}/{len(param_grid)}: gamma=({gamma[0]:.4g}, {gamma[1]:.4g}), "
            f"r_C={r_C:.4g}, mass_weight={mass_weight:.4g}, "
            f"elapsed={(time.time() - t0):.1f}s, eta={((time.time() - t0) / (p + 1) * (len(param_grid) - (p + 1))):.1f}s",
            flush=True,
        )

        for i_subset, idx in enumerate(sample_subset, start=1):
            if (i_subset == 1) or (i_subset % 10 == 0) or (i_subset == len(sample_subset)):
                elapsed_subset = time.time() - set_start
                eta_subset = (elapsed_subset / i_subset) * (len(sample_subset) - i_subset)
                print(
                    f"[grid]   subset idx={i_subset}/{len(sample_subset)}, "
                    f"elapsed={elapsed_subset:.1f}s, eta={eta_subset:.1f}s",
                    flush=True,
                )
            args_tuple = (
                x_init_growth[idx],
                no3_growth[idx],
                time_growth[idx],
                x_final_growth[idx],
                no3_chl[idx],
                time_chl[idx],
                gamma,
                mass_weight,
                r_C,
                False,
            )

            result = differential_evolution(
                calculate_error,
                bounds=bounds,
                args=args_tuple,
                strategy='best1bin',
                maxiter=de_maxiter,
                popsize=de_popsize,
                tol=de_tol,
                polish=True,
            )

            _, _, _, _, _, mse_growth, mse_chl, mse_biomass, _ = calculate_error(
                result.x, *args_tuple[:-1], return_full_trajectories=True
            )

            rows.append({
                'param_set': p,
                'sample_idx': int(idx),
                'gamma0': float(gamma[0]),
                'gamma1': float(gamma[1]),
                'r_C': float(r_C),
                'mass_weight': float(mass_weight),
                'total_error': float(result.fun),
                'mse_growth': float(mse_growth),
                'mse_chl': float(mse_chl),
                'mse_biomass0': float(mse_biomass[0]),
                'mse_biomass1': float(mse_biomass[1]),
            })

    elapsed_grid_total = time.time() - t0
    print(f"[grid] complete: elapsed={elapsed_grid_total:.1f}s", flush=True)

    return pd.DataFrame(rows), sample_subset, param_grid


def summarize_grid(grid_search_errors, gamma0_grid, gamma1_grid, rC_grid, mass_weight_grid):
    avg_errors = grid_search_errors.groupby('param_set').mean(numeric_only=True).reset_index()
    std_errors = grid_search_errors.groupby('param_set').std(numeric_only=True).reset_index()

    shape = (len(gamma0_grid), len(gamma1_grid), len(rC_grid), len(mass_weight_grid))

    total_errors_array = np.array(avg_errors['total_error']).reshape(shape)
    mse_growth_array = np.array(avg_errors['mse_growth']).reshape(shape)
    mse_biomass_0_array = np.array(avg_errors['mse_biomass0']).reshape(shape)
    mse_biomass_1_array = np.array(avg_errors['mse_biomass1']).reshape(shape)

    mse_growth_std_array = np.array(std_errors['mse_growth']).reshape(shape)
    mse_biomass_0_std_array = np.array(std_errors['mse_biomass0']).reshape(shape)
    mse_biomass_1_std_array = np.array(std_errors['mse_biomass1']).reshape(shape)

    return {
        'avg_errors': avg_errors,
        'std_errors': std_errors,
        'total_errors_array': total_errors_array,
        'mse_growth_array': mse_growth_array,
        'mse_biomass_0_array': mse_biomass_0_array,
        'mse_biomass_1_array': mse_biomass_1_array,
        'mse_growth_std_array': mse_growth_std_array,
        'mse_biomass_0_std_array': mse_biomass_0_std_array,
        'mse_biomass_1_std_array': mse_biomass_1_std_array,
    }


def run_fixed_hparams_fullfit(data, gamma, r_C, mass_weight,
                              de_maxiter, de_popsize, de_tol,
                              n_interp=100):
    no3_growth = data['no3_growth']
    no3_chl = data['no3_chl']
    x_init_growth = data['x_init_growth']
    x_final_growth = data['x_final_growth']
    time_growth = data['time_growth']
    time_chl = data['time_chl']

    num_samples = len(no3_growth)
    bounds = [(0.0, 5.0), (0.0, 5.0), (0.0, 4.0)]

    final_params = np.zeros((num_samples, 3))
    errors = []

    A_growth_preds = np.zeros((num_samples, no3_growth.shape[1]))
    C_growth_preds = np.zeros((num_samples, no3_growth.shape[1]))
    x_growth_preds = np.zeros((num_samples, 2, no3_growth.shape[1]))

    A_growth_exact_interp = np.zeros((num_samples, n_interp))
    C_growth_exact_interp = np.zeros((num_samples, n_interp))
    x_growth_exact_interp = np.zeros((num_samples, 2, n_interp))

    print(f'[fullfit] running on {num_samples} samples', flush=True)
    t_fullfit = time.time()
    for i in range(num_samples):
        if (i % 25 == 0) or (i == num_samples - 1):
            elapsed_fullfit = time.time() - t_fullfit
            eta_fullfit = (elapsed_fullfit / (i + 1)) * (num_samples - (i + 1))
            print(f"[fullfit] sample {i + 1}/{num_samples}, elapsed={elapsed_fullfit:.1f}s, eta={eta_fullfit:.1f}s", flush=True)
        args_tuple = (
            x_init_growth[i],
            no3_growth[i],
            time_growth[i],
            x_final_growth[i],
            no3_chl[i],
            time_chl[i],
            gamma,
            mass_weight,
            r_C,
            False,
        )

        result = differential_evolution(
            calculate_error,
            bounds=bounds,
            args=args_tuple,
            strategy='best1bin',
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=de_tol,
            polish=True,
        )

        final_params[i, :] = result.x

        A_growth_model, _, x_pred_growth, _, C, mse_growth, mse_chl, mse_biomass, _ = calculate_error(
            result.x,
            *args_tuple[:-1],
            return_full_trajectories=True,
        )

        A_growth_preds[i, :] = A_growth_model
        C_growth_preds[i, :] = C
        x_growth_preds[i, :, :] = x_pred_growth

        errors.append({
            'total_error': float(result.fun),
            'mse_growth': float(mse_growth),
            'mse_chl': float(mse_chl),
            'mse_biomass0': float(mse_biomass[0]),
            'mse_biomass1': float(mse_biomass[1]),
        })

        t_interp = np.linspace(0, time_growth[i][-1], n_interp)
        A_growth_interp, C_growth_interp, x_growth_interp, _, _, _ = integrate_eom(
            result.x,
            no3_chl[i][0],
            no3_growth[i][0],
            x_init_growth[i],
            t_interp,
            time_chl[i],
            gamma,
            C0=C[0],
            r_C=r_C,
        )

        A_growth_exact_interp[i, :] = A_growth_interp
        C_growth_exact_interp[i, :] = C_growth_interp
        x_growth_exact_interp[i, :, :] = x_growth_interp

    return {
        'final_params': final_params,
        'errors_df': pd.DataFrame(errors),
        'A_growth_preds': A_growth_preds,
        'C_growth_preds': C_growth_preds,
        'x_growth_preds': x_growth_preds,
        'A_growth_exact_interp': A_growth_exact_interp,
        'C_growth_exact_interp': C_growth_exact_interp,
        'x_growth_exact_interp': x_growth_exact_interp,
    }


def main():
    t_main = time.time()
    ap = argparse.ArgumentParser()

    ap.add_argument('--save_dir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='auto')

    ap.add_argument('--masterroot', type=str,
                    default='/project/vitelli/matthew/soil_microbiome_model_reduction/prediction_models/soils')
    ap.add_argument('--subroots', type=str, default='')
    ap.add_argument('--output_name', type=str, default='no3')
    ap.add_argument('--tax_name', type=str, default='ASV')
    ap.add_argument('--n_keep', type=int, default=5)
    ap.add_argument('--ensemble_rank', type=int, default=1)

    ap.add_argument('--gamma0_min', type=float, required=True)
    ap.add_argument('--gamma0_max', type=float, required=True)
    ap.add_argument('--gamma0_n', type=int, required=True)

    ap.add_argument('--gamma1_min', type=float, required=True)
    ap.add_argument('--gamma1_max', type=float, required=True)
    ap.add_argument('--gamma1_n', type=int, required=True)

    ap.add_argument('--rC_min', type=float, default=1.0)
    ap.add_argument('--rC_max', type=float, default=1.0)
    ap.add_argument('--rC_n', type=int, default=1)

    ap.add_argument('--mass_weight_min', type=float, required=True)
    ap.add_argument('--mass_weight_max', type=float, required=True)
    ap.add_argument('--mass_weight_n', type=int, required=True)

    ap.add_argument('--sample_subset_size', type=int, default=20)

    ap.add_argument('--de_popsize', type=int, default=10)
    ap.add_argument('--de_maxiter', type=int, default=1000)
    ap.add_argument('--de_tol', type=float, default=1e-6)

    ap.add_argument('--fit_full_dataset', type=int, default=0)
    ap.add_argument('--gamma_fixed', type=str, default='0.4,2.4')
    ap.add_argument('--rC_fixed', type=float, default=1.0)
    ap.add_argument('--mass_weight_fixed', type=float, default=10.0)

    args = ap.parse_args()
    print(args, flush=True)
    print('[config] differential_evolution strategy fixed to best1bin, polish fixed to True', flush=True)

    if args.device == 'auto':
        compute_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cuda':
        compute_device = 'cuda:0'
    else:
        compute_device = 'cpu'
    print(f'[config] compute device: {compute_device}', flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    subroots = parse_subroots(args.subroots)
    if len(subroots) == 0:
        subroots = default_subroots()

    gamma0_grid = np.logspace(np.log10(args.gamma0_min), np.log10(args.gamma0_max), args.gamma0_n)
    gamma1_grid = np.logspace(np.log10(args.gamma1_min), np.log10(args.gamma1_max), args.gamma1_n)
    rC_grid = np.logspace(np.log10(args.rC_min), np.log10(args.rC_max), args.rC_n)
    mass_weight_grid = np.logspace(np.log10(args.mass_weight_min), np.log10(args.mass_weight_max), args.mass_weight_n)

    print(f"[main] entering data load, elapsed={(time.time() - t_main):.1f}s", flush=True)

    loss_df = load_loss_df(args.masterroot, subroots, output_name=args.output_name, tax_name=args.tax_name, device=compute_device)

    print(f"[main] data load finished, elapsed={(time.time() - t_main):.1f}s", flush=True)

    data = prepare_group_data(
        loss_df,
        n_keep=args.n_keep,
        output_name=args.output_name,
        ensemble_rank=args.ensemble_rank,
        device=compute_device,
    )

    print(f"[main] group preparation finished, elapsed={(time.time() - t_main):.1f}s", flush=True)

    grid_search_errors, sample_subset, param_grid = run_grid_search(
        data,
        gamma0_grid,
        gamma1_grid,
        rC_grid,
        mass_weight_grid,
        args.sample_subset_size,
        args.de_maxiter,
        args.de_popsize,
        args.de_tol,
        seed=args.seed,
    )

    print(f"[main] grid search finished, elapsed={(time.time() - t_main):.1f}s", flush=True)

    summary = summarize_grid(grid_search_errors, gamma0_grid, gamma1_grid, rC_grid, mass_weight_grid)

    fullfit_payload = None

    if args.fit_full_dataset:
        gamma_fixed = np.asarray(parse_float_list(args.gamma_fixed))

        fullfit = run_fixed_hparams_fullfit(
            data,
            gamma=gamma_fixed,
            r_C=args.rC_fixed,
            mass_weight=args.mass_weight_fixed,
            de_maxiter=args.de_maxiter,
            de_popsize=args.de_popsize,
            de_tol=args.de_tol,
        )

        fullfit_payload = {
            'final_params': fullfit['final_params'],
            'errors_df': fullfit['errors_df'],
            'A_growth_preds': fullfit['A_growth_preds'],
            'C_growth_preds': fullfit['C_growth_preds'],
            'x_growth_preds': fullfit['x_growth_preds'],
            'A_growth_exact_interp': fullfit['A_growth_exact_interp'],
            'C_growth_exact_interp': fullfit['C_growth_exact_interp'],
            'x_growth_exact_interp': fullfit['x_growth_exact_interp'],
        }

        print(f"[main] fullfit finished, elapsed={(time.time() - t_main):.1f}s", flush=True)

    payload = {
        'args': vars(args),
        'subroots': subroots,
        'gamma0_grid': np.asarray(gamma0_grid),
        'gamma1_grid': np.asarray(gamma1_grid),
        'rC_grid': np.asarray(rC_grid),
        'mass_weight_grid': np.asarray(mass_weight_grid),
        'loss_df': loss_df,
        'sample_subset': sample_subset,
        'grid_search_errors': grid_search_errors,
        'grid_avg_errors': summary['avg_errors'],
        'grid_std_errors': summary['std_errors'],
        'total_errors_array': summary['total_errors_array'],
        'mse_growth_array': summary['mse_growth_array'],
        'mse_biomass_0_array': summary['mse_biomass_0_array'],
        'mse_biomass_1_array': summary['mse_biomass_1_array'],
        'mse_growth_std_array': summary['mse_growth_std_array'],
        'mse_biomass_0_std_array': summary['mse_biomass_0_std_array'],
        'mse_biomass_1_std_array': summary['mse_biomass_1_std_array'],
        'x_init_growth': data['x_init_growth'],
        'x_final_growth': data['x_final_growth'],
        'no3_growth': data['no3_growth'],
        'no3_chl': data['no3_chl'],
        'time_growth': data['time_growth'],
        'time_chl': data['time_chl'],
        'ensembles_to_keep': data['ensembles_to_keep'],
        'chosen_ens': data['chosen_ens'],
        'fullfit': fullfit_payload,
        'compute_device': compute_device,
    }

    np.savez_compressed(
        os.path.join(args.save_dir, 'fig5_hparam_search_results.npz'),
        payload=np.asarray([payload], dtype=object),
    )

    print(f"Done. Outputs written to: {args.save_dir}", flush=True)
    print(f"[main] total runtime, elapsed={(time.time() - t_main):.1f}s", flush=True)


if __name__ == '__main__':
    main()
