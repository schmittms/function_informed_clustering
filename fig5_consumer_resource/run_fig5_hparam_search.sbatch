#!/bin/bash
#SBATCH --output=./out/hparam_search.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=vitelli
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1

module load python/anaconda-2022.05
source activate /project/vitelli/matthew/torch_venv

#SBATCH --gres=gpu:1


cd /home/schmittms/project_vitelli_matthew/scifi_public/fig5_consumer_resource

seed=0

# Data/model selection
masterroot='/project/vitelli/matthew/soil_microbiome_model_reduction/prediction_models/soils'
subroots=''   # semicolon-separated list; keep empty to use defaults from notebook
output_name='no3'
tax_name='ASV'
n_keep=5
ensemble_rank=1

# Hyperparameter ranges (logspace: min, max, number of values)
gamma0_min=0.01
gamma0_max=2.0
gamma0_n=8

gamma1_min=0.5
gamma1_max=8.0
gamma1_n=8

rC_min=1.0
rC_max=1.0
rC_n=1

mass_weight_min=0.1
mass_weight_max=100.0
mass_weight_n=4

# Grid search controls
sample_subset_size=20
de_popsize=15
de_maxiter=1000
de_tol=1e-6

# Optional full-dataset fit for fixed hparams (helper output for later notebook plots)
fit_full_dataset=1
gamma_fixed='0.21,2.44'
rC_fixed=1.0
mass_weight_fixed=10.0

save_dir=$(date +./out/fig5_hparam_%y%m%d_%H%M%S)_seed-${seed}
mkdir -p "$save_dir"

echo "Saving outputs to: $save_dir"

python run_fig5_hparam_search.py \
    --save_dir "$save_dir" \
    --seed $seed \
    --masterroot "$masterroot" \
    --subroots "$subroots" \
    --output_name "$output_name" \
    --tax_name "$tax_name" \
    --n_keep $n_keep \
    --ensemble_rank $ensemble_rank \
    --gamma0_min $gamma0_min \
    --gamma0_max $gamma0_max \
    --gamma0_n $gamma0_n \
    --gamma1_min $gamma1_min \
    --gamma1_max $gamma1_max \
    --gamma1_n $gamma1_n \
    --rC_min $rC_min \
    --rC_max $rC_max \
    --rC_n $rC_n \
    --mass_weight_min $mass_weight_min \
    --mass_weight_max $mass_weight_max \
    --mass_weight_n $mass_weight_n \
    --sample_subset_size $sample_subset_size \
    --de_popsize $de_popsize \
    --de_maxiter $de_maxiter \
    --de_tol $de_tol \
    --fit_full_dataset $fit_full_dataset \
    --gamma_fixed "$gamma_fixed" \
    --rC_fixed $rC_fixed \
    --mass_weight_fixed $mass_weight_fixed
