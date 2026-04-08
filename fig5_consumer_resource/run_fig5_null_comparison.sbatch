#!/bin/bash
#SBATCH --output=./out/null_models.out
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

# Non-null model gamma (from notebook)
gamma_nonnull='0.21,2.44'

# Null model gamma choice:
#   fixed   -> use gamma_null_fixed
#   median2 -> use 2*median(log(x_final/x_init)) per null realization
gamma_null_mode='fixed' 
gamma_null_fixed='0.21,2.44'

# Which biomass group(s) to randomize in null construction: both | biomass1 | biomass2
null_shuffle_target='both'

# Optimization and null controls
mass_weight=10.0
rC=1.0
n_nulls=200
subset_size=100

de_popsize=15
de_maxiter=1000
de_tol=1e-6

save_dir=$(date +./out/null_modeling_%y%m%d_%H%M%S)_s-${seed}_nm-${gamma_null_mode}_ns-${null_shuffle_target}_ss-${subset_size}
mkdir -p "$save_dir"

echo "Saving outputs to: $save_dir"

python run_fig5_null_comparison.py \
    --save_dir "$save_dir" \
    --seed $seed \
    --masterroot "$masterroot" \
    --subroots "$subroots" \
    --output_name "$output_name" \
    --tax_name "$tax_name" \
    --n_keep $n_keep \
    --ensemble_rank $ensemble_rank \
    --gamma_nonnull "$gamma_nonnull" \
    --gamma_null_mode "$gamma_null_mode" \
    --gamma_null_fixed "$gamma_null_fixed" \
    --null_shuffle_target "$null_shuffle_target" \
    --mass_weight $mass_weight \
    --rC $rC \
    --n_nulls $n_nulls \
    --subset_size $subset_size \
    --de_popsize $de_popsize \
    --de_maxiter $de_maxiter \
    --de_tol $de_tol
