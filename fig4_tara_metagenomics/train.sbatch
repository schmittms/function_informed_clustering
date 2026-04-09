#!/bin/bash
#SBATCH --output=./out/output_%a.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-47
#SBATCH --nodes=1
#SBATCH --time=3:45:00

module load python/anaconda-2022.05
source activate /project/vitelli/matthew/torch_venv
cd /home/schmittms/project_vitelli_matthew/scifi_public/fig4_tara_metagenomics
#SBATCH --gres=gpu:1

idx=$SLURM_ARRAY_TASK_ID

n_steps=30000
save_freq=100
num_workers=0
N_ensemble=15
n_layers=2
act=gelu
nchannels=128
seed=$((1000 + idx))
schedule_rate=0.9999
opt_method=adam
dropout_input=0.0

n=0
#for x in {0.,0.05}" "{1e-2,1e-3,}" "{1e-2,1e-3,}" "{1e-1,1e-2}" "{2.,}" "{0.9999,}" "{0.1,}" "{'nitrate,Oxygen,Temperature','nitrate','Oxygen','Temperature','nitrate,Temperature','nitrate,Oxygen'};
for x in {0.,0.05,0.1}" "{1e-2,1e-3,}" "{1e-2,1e-3,}" "{1e-1,1e-2}" "{2.,}" "{0.9999,}" "{0.1,}" "{'nitrate,Oxygen,Temperature','nitrate',};
do
    ar=($x)
    N=${#ar[@]}
    if [[ "$N" == "8" ]]; then
	    if [[ "$n" == "$idx" ]]; then
		dropout=${ar[0]}
		lr=${ar[1]}
		gate_lr=${ar[2]}
		beta=${ar[3]}
		tau_init=${ar[4]}
		tau_relax_rate=${ar[5]}
		tau_min=${ar[6]}
		targets=${ar[7]}
	    fi
	    n=$((n+1))
    fi
done

echo "TOTAL JOBS: $n"

save_dir=$(date +./trained_models/gatedgumbel_%y%m%d_%H%M%S)_targets-${targets}_N-${nchannels}_L-${n_layers}_DO-${dropout}_LR-${lr}_gateLR-${gate_lr}_beta-${beta}_TAU-${tau_init}_${tau_relax_rate}_${tau_min}_ACT-${act}_NENS-${N_ensemble}_Nsteps-${n_steps}_${idx}

echo $save_dir

python train.py \
	--n_steps $n_steps \
	--save_freq $save_freq \
	--save_dir $save_dir \
	--num_workers $num_workers \
	--N_ensemble $N_ensemble \
	--seed $seed \
	--n_layers $n_layers \
	--act $act \
	--dropout $dropout \
	--dropout_input $dropout_input \
	--nchannels $nchannels \
	--tau_init $tau_init \
	--tau_relax_rate $tau_relax_rate \
	--tau_min $tau_min \
	--lr $lr \
	--gate_lr $gate_lr \
	--beta $beta \
	--schedule_rate $schedule_rate \
	--opt_method $opt_method \
	--targets $targets
