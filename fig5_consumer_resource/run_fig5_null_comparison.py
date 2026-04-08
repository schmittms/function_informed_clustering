import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.optimize import differential_evolution

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import GatedModel
from dataset import SoilDatasetFiltered
from run_fig5_hparam_search import (
    calculate_error,
    parse_subroots,
    default_subroots,
    load_loss_df,
    prepare_group_data,
)


def parse_float_list(spec: str):
    spec = spec.strip()
    if not spec:
        return []
    return [float(x) for x in spec.split(',') if x.strip()]


def build_real_and_null_groups(dataset_train, x_init_growth, x_final_growth, n_nulls, seed=0, null_shuffle_target='both'):
    rng = np.random.default_rng(seed)

    avg_abd = np.mean(x_init_growth, axis=0)

    CHL_mask = dataset_train.df.Chloramphenicol != 0
    abd_init = dataset_train.comp_t0_asv
    abd_end = dataset_train.comp_tT_asv
    abd_end_chl_pos = abd_end[CHL_mask]

    asv_idx_perm = np.asarray([rng.permutation(abd_end_chl_pos.shape[-1]) for _ in range(n_nulls)])

    mean_abd_chl_pos = np.mean(abd_end_chl_pos, axis=0)

    fake_group_init_abundances = []
    fake_group_end_abundances = []
    for perm in asv_idx_perm:
        permuted_abd = mean_abd_chl_pos[perm]
        cumsum = np.cumsum(permuted_abd)

        group_1_idx = next(i for i, v in enumerate(cumsum) if v > avg_abd[0])
        group_2_idx = next(i + group_1_idx for i, v in enumerate(cumsum[group_1_idx:]) if v > (avg_abd[0] + avg_abd[1]))

        fake_group_init_abundances.append([
            np.sum(abd_init[:, perm[:group_1_idx]], axis=1),
            np.sum(abd_init[:, perm[group_1_idx:group_2_idx]], axis=1),
        ])
        fake_group_end_abundances.append([
            np.sum(abd_end[:, perm[:group_1_idx]], axis=1),
            np.sum(abd_end[:, perm[group_1_idx:group_2_idx]], axis=1),
        ])

    fake_group_init_abundances = np.asarray(fake_group_init_abundances).swapaxes(-1, -2)
    fake_group_end_abundances = np.asarray(fake_group_end_abundances).swapaxes(-1, -2)

    real_group_init_abundances = np.tile(x_init_growth[None, :, :], (n_nulls, 1, 1))
    real_group_end_abundances = np.tile(x_final_growth[None, :, :], (n_nulls, 1, 1))

    fake_group_end_abundances = fake_group_end_abundances[:, ~CHL_mask, :]
    fake_group_init_abundances = fake_group_init_abundances[:, ~CHL_mask, :]

    if null_shuffle_target == 'biomass1':
        fake_group_init_abundances[:, :, 1] = real_group_init_abundances[:, :, 1]
        fake_group_end_abundances[:, :, 1] = real_group_end_abundances[:, :, 1]
    elif null_shuffle_target == 'biomass2':
        fake_group_init_abundances[:, :, 0] = real_group_init_abundances[:, :, 0]
        fake_group_end_abundances[:, :, 0] = real_group_end_abundances[:, :, 0]

    return {
        'fake_group_init_abundances': fake_group_init_abundances,
        'fake_group_end_abundances': fake_group_end_abundances,
        'real_group_init_abundances': real_group_init_abundances,
        'real_group_end_abundances': real_group_end_abundances,
    }


def run_null_comparison(data,
                        null_groups,
                        gamma_nonnull,
                        gamma_null_mode,
                        gamma_null_fixed,
                        mass_weight,
                        r_C,
                        subset_size,
                        de_maxiter,
                        de_popsize,
                        de_tol,
                        seed=0):
    rng = np.random.default_rng(seed)

    no3_growth = data['no3_growth']
    no3_chl = data['no3_chl']
    time_growth = data['time_growth']
    time_chl = data['time_chl']

    fake_group_init_abundances = null_groups['fake_group_init_abundances']
    fake_group_end_abundances = null_groups['fake_group_end_abundances']
    real_group_init_abundances = null_groups['real_group_init_abundances']
    real_group_end_abundances = null_groups['real_group_end_abundances']

    bounds = [(0.0, 5.0), (0.0, 5.0), (0.0, 4.0)]

    biomass_errors_ensemble = {'Fake': [], 'Real': []}
    nitrate_errors_ensemble = {'Fake': [], 'Real': []}
    sample_subsets = []
    gamma_used = {'Fake': [], 'Real': []}

    n_nulls = len(fake_group_init_abundances)
    t_null = time.time()
    print(f'[null] total null models: {n_nulls}', flush=True)
    print(f'[null] subset size per model: {subset_size}', flush=True)

    for i in range(n_nulls):
        t_model = time.time()
        sample_subset = rng.choice(np.arange(no3_growth.shape[0]), size=subset_size, replace=False)
        sample_subsets.append(sample_subset)
        elapsed_null = time.time() - t_null
        eta_null = (elapsed_null / (i + 1)) * (n_nulls - (i + 1))
        print(f"[null] model {i + 1}/{n_nulls}, elapsed={elapsed_null:.1f}s, eta={eta_null:.1f}s", flush=True)

        for label, x_init_growth, x_final_growth in [
            ('Fake', fake_group_init_abundances[i], fake_group_end_abundances[i]),
            ('Real', real_group_init_abundances[i], real_group_end_abundances[i]),
        ]:
            if label == 'Real':
                gamma_eff = gamma_nonnull
            else:
                if gamma_null_mode == 'fixed':
                    gamma_eff = gamma_null_fixed
                else:
                    gamma_eff = 2 * np.median(np.log((x_final_growth + 1e-3) / (x_init_growth + 1e-3)), axis=0)

            gamma_used[label].append(np.asarray(gamma_eff))

            biomass_sample_errors = []
            nitrate_sample_errors = []

            for j, idx in enumerate(sample_subset, start=1):
                if (j == 1) or (j % 10 == 0) or (j == len(sample_subset)):
                    elapsed_model = time.time() - t_model
                    eta_model = (elapsed_model / j) * (len(sample_subset) - j)
                    print(
                        f"[null]   label={label} subset idx={j}/{len(sample_subset)}, "
                        f"elapsed={elapsed_model:.1f}s, eta={eta_model:.1f}s",
                        flush=True,
                    )
                args_tuple = (
                    x_init_growth[idx],
                    no3_growth[idx],
                    time_growth[idx],
                    x_final_growth[idx],
                    no3_chl[idx],
                    time_chl[idx],
                    gamma_eff,
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

                _, _, _, _, _, mse_growth, _, mse_biomass, _ = calculate_error(
                    result.x,
                    *args_tuple[:-1],
                    return_full_trajectories=True,
                )

                biomass_sample_errors.append(mse_biomass)
                nitrate_sample_errors.append(mse_growth)

            biomass_errors_ensemble[label].append(biomass_sample_errors)
            nitrate_errors_ensemble[label].append(nitrate_sample_errors)
        elapsed_model_total = time.time() - t_model
        print(f"[null]   finished model {i + 1}/{n_nulls}, elapsed={elapsed_model_total:.1f}s", flush=True)

    elapsed_null_total = time.time() - t_null
    print(f"[null] complete, elapsed={elapsed_null_total:.1f}s", flush=True)

    biomass_errors_ensemble['Fake'] = np.asarray(biomass_errors_ensemble['Fake'])
    biomass_errors_ensemble['Real'] = np.asarray(biomass_errors_ensemble['Real'])
    nitrate_errors_ensemble['Fake'] = np.asarray(nitrate_errors_ensemble['Fake'])
    nitrate_errors_ensemble['Real'] = np.asarray(nitrate_errors_ensemble['Real'])

    return {
        'biomass_errors_ensemble': biomass_errors_ensemble,
        'nitrate_errors_ensemble': nitrate_errors_ensemble,
        'sample_subsets': np.asarray(sample_subsets),
        'gamma_used': {
            'Fake': np.asarray(gamma_used['Fake']),
            'Real': np.asarray(gamma_used['Real']),
        },
    }


def compute_probabilities_and_pvals(results):
    fake_bio = results['biomass_errors_ensemble']['Fake']
    real_bio = results['biomass_errors_ensemble']['Real']
    fake_nit = results['nitrate_errors_ensemble']['Fake']
    real_nit = results['nitrate_errors_ensemble']['Real']

    fake_mean_bio = np.mean(fake_bio, axis=1)
    real_mean_bio = np.mean(real_bio, axis=1)
    fake_mean_nit = np.mean(fake_nit, axis=1)
    real_mean_nit = np.mean(real_nit, axis=1)

    prob_biomass0_better = float(np.mean(fake_mean_bio[:, 0] < np.mean(real_mean_bio[:, 0])))
    prob_biomass1_better = float(np.mean(fake_mean_bio[:, 1] < np.mean(real_mean_bio[:, 1])))
    prob_nitrate_better = float(np.mean(fake_mean_nit < np.mean(real_mean_nit)))

    p_bio0 = float((np.sum(fake_mean_bio[:, 0] <= np.mean(real_mean_bio[:, 0])) + 1) / (len(fake_mean_bio) + 1))
    p_bio1 = float((np.sum(fake_mean_bio[:, 1] <= np.mean(real_mean_bio[:, 1])) + 1) / (len(fake_mean_bio) + 1))
    p_nitrate = float((np.sum(fake_mean_nit <= np.mean(real_mean_nit)) + 1) / (len(fake_mean_nit) + 1))

    return {
        'prob_biomass0_better': prob_biomass0_better,
        'prob_biomass1_better': prob_biomass1_better,
        'prob_nitrate_better': prob_nitrate_better,
        'pvalue_biomass0': p_bio0,
        'pvalue_biomass1': p_bio1,
        'pvalue_nitrate': p_nitrate,
        'mean_real_biomass0': float(np.mean(real_mean_bio[:, 0])),
        'mean_real_biomass1': float(np.mean(real_mean_bio[:, 1])),
        'mean_real_nitrate': float(np.mean(real_mean_nit)),
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

    ap.add_argument('--gamma_nonnull', type=str, default='0.4,2.4')
    ap.add_argument('--gamma_null_mode', type=str, default='fixed', choices=['fixed', 'median2'])
    ap.add_argument('--gamma_null_fixed', type=str, default='0.4,2.4')
    ap.add_argument('--null_shuffle_target', type=str, default='both', choices=['both', 'biomass1', 'biomass2'])

    ap.add_argument('--mass_weight', type=float, default=10.0)
    ap.add_argument('--rC', type=float, default=1.0)

    ap.add_argument('--n_nulls', type=int, default=200)
    ap.add_argument('--subset_size', type=int, default=100)

    ap.add_argument('--de_popsize', type=int, default=10)
    ap.add_argument('--de_maxiter', type=int, default=1000)
    ap.add_argument('--de_tol', type=float, default=1e-6)

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

    gamma_nonnull = np.asarray(parse_float_list(args.gamma_nonnull))
    gamma_null_fixed = np.asarray(parse_float_list(args.gamma_null_fixed))

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

    null_groups = build_real_and_null_groups(
        dataset_train=data['dataset_train'],
        x_init_growth=data['x_init_growth'],
        x_final_growth=data['x_final_growth'],
        n_nulls=args.n_nulls,
        seed=args.seed,
        null_shuffle_target=args.null_shuffle_target,
    )

    print(f"[main] null group construction finished, elapsed={(time.time() - t_main):.1f}s", flush=True)

    results = run_null_comparison(
        data,
        null_groups,
        gamma_nonnull=gamma_nonnull,
        gamma_null_mode=args.gamma_null_mode,
        gamma_null_fixed=gamma_null_fixed,
        mass_weight=args.mass_weight,
        r_C=args.rC,
        subset_size=args.subset_size,
        de_maxiter=args.de_maxiter,
        de_popsize=args.de_popsize,
        de_tol=args.de_tol,
        seed=args.seed,
    )

    print(f"[main] null comparison finished, elapsed={(time.time() - t_main):.1f}s", flush=True)

    summary = compute_probabilities_and_pvals(results)

    payload = {
        'args': vars(args),
        'subroots': subroots,
        'gamma_nonnull': gamma_nonnull,
        'gamma_null_fixed': gamma_null_fixed,
        'loss_df': loss_df,
        'null_groups': null_groups,
        'results': results,
        'summary': summary,
        'x_init_growth': data['x_init_growth'],
        'x_final_growth': data['x_final_growth'],
        'no3_growth': data['no3_growth'],
        'no3_chl': data['no3_chl'],
        'time_growth': data['time_growth'],
        'time_chl': data['time_chl'],
        'ensembles_to_keep': data['ensembles_to_keep'],
        'chosen_ens': data['chosen_ens'],
        'compute_device': compute_device,
    }

    np.savez_compressed(
        os.path.join(args.save_dir, 'fig5_null_comparison_results.npz'),
        payload=np.asarray([payload], dtype=object),
    )

    print(f'Done. Outputs written to: {args.save_dir}', flush=True)
    print('Summary:', summary, flush=True)
    print(
        f"p-values: biomass0={summary['pvalue_biomass0']:.6g}, "
        f"biomass1={summary['pvalue_biomass1']:.6g}, "
        f"nitrate={summary['pvalue_nitrate']:.6g}",
        flush=True,
    )
    print(f"[main] total runtime, elapsed={(time.time() - t_main):.1f}s", flush=True)


if __name__ == '__main__':
    main()
