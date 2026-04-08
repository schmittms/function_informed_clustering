#!/bin/bash
#SBATCH --output=./out/output_%a.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=12
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load python/anaconda-2022.05
source activate /project/vitelli/matthew/torch_venv
cd /home/schmittms/project_vitelli_matthew/scifi_public/fig3_soil

idx=$SLURM_ARRAY_TASK_ID

n_steps=50000
save_freq=50
num_workers=0
taxa='Phylum'
N_ensemble=15
n_layers=3
act=gelu
nchannels=128
seed=$((1000 + idx))
early_stopping_limit=10000
schedule_rate=0.9999
opt_method=adam

n=0
for x in {0.,0.05,0.1}" "{1e-3,}" "{6,}" "{2.,1.}" "{0.9999,}" "{0.1,}" "{neg,both};
do
    ar=($x)
    N=${#ar[@]}
    if [[ "$N" == "7" ]]; then
	    if [[ "$n" == "$idx" ]]; then
		dropout=${ar[0]}
		lr=${ar[1]}
		N_clusters=${ar[2]}
		tau_init=${ar[3]}
		tau_relax_rate=${ar[4]}
		tau_min=${ar[5]}
		CHL=${ar[6]}
	    fi
	    n=$((n+1))
    fi
done

echo "TOTAL JOBS: $n"

save_dir=$(date +./trained_models/gumbel_%y%m%d_%H%M%S)_N-${nchannels}_L-${n_layers}_predict_Nclusts${N_clusters}_${idx}

echo $save_dir

python train.py \
	--n_steps $n_steps \
	--save_dir $save_dir \
	--num_workers $num_workers \
	--CHL $CHL \
	--seed $seed \
	--taxa $taxa \
	--N_clusters $N_clusters \
	--N_ensemble $N_ensemble \
	--n_layers $n_layers \
	--act $act \
	--dropout $dropout \
	--nchannels $nchannels \
	--tau_init $tau_init \
	--tau_relax_rate $tau_relax_rate \
	--tau_min $tau_min \
	--lr $lr \
	--schedule_rate $schedule_rate \
	--opt_method $opt_method \
	--save_freq $save_freq \
	--early_stopping_limit $early_stopping_limit
