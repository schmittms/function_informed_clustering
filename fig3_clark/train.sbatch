#!/bin/bash
#SBATCH --output=./out/output_%a.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=vitelli-gpu
#SBATCH --ntasks-per-node=12
#SBATCH --array=-23
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load python/anaconda-2022.05
source activate /project/vitelli/matthew/torch_venv
cd /home/schmittms/project_vitelli_matthew/scifi_public/fig3_clark

idx=$SLURM_ARRAY_TASK_ID

test_train_frac=0.8
n_steps=10000
save_freq=100
num_workers=0
N_ensemble=15
n_layers=2
act=gelu
nchannels=256
seed=$((1000 + idx))
dropout_input=0.
schedule_rate=0.999
opt_method=adam

n=0
for x in {0.,0.05}" "{1e-2,1e-3}" "{1.,}" "{0.999,0.9995}" "{0.1,}" "{Butyrate,Acetate,Succinate};
do
    ar=($x)
    N=${#ar[@]}
    if [[ "$N" == "6" ]]; then
	    if [[ "$n" == "$idx" ]]; then
		dropout=${ar[0]}
		lr=${ar[1]}
		tau_init=${ar[2]}
		tau_relax_rate=${ar[3]}
		tau_min=${ar[4]}
		output=${ar[5]}
	    fi
	    n=$((n+1))
    fi
done

echo "TOTAL JOBS: $n"

save_dir=$(date +./trained_models/gumbel_%y%m%d_%H%M%S)_N-${nchannels}_L-${n_layers}_OUT-${output}_${idx}

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
	--test_train_frac $test_train_frac \
	--output $output \
	--nchannels $nchannels \
	--tau_init $tau_init \
	--tau_relax_rate $tau_relax_rate \
	--tau_min $tau_min \
	--lr $lr \
	--schedule_rate $schedule_rate \
	--opt_method $opt_method
