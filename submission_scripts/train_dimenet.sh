#!/bin/bash
#SBATCH -J dnpp_xtb_s1.log
#SBATCH -o ./logs/dnpp_xtb_s1.log
##SBATCH -t 1000:00:00
#SBATCH -p xeon-g6-volta
#####SBATCH -p xeon-p8
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2

source /etc/profile
source activate ConfSolv
export PYTHONPATH=$PYTHONPATH:$(pwd)

CODE=~/Projects/conf_solv
log_dir=$CODE/exps/2023_08_21/seed1_split0
coords_path=$CODE/data/full/dft_coords.pkl.gz
energies_path=$CODE/data/full/free_energies.pkl.gz
split_path=$CODE/data/full/splits/scaffold_split0.pkl

num_workers=20
n_epochs=100
batch_size=64
lr=5e-4
weight_decay=0
hidden_channels=64
num_blocks=3
solute_model=DimeNetPP
max_confs=5
seed=1

echo "Start time: $(date '+%Y-%m-%d_%H:%M:%S')"
python -u $CODE/conf_solv/train.py \
       --coords_path $coords_path \
       --energies_path $energies_path \
       --gpus 2 \
       --log_dir $log_dir \
       --split_path $split_path \
       --n_epochs $n_epochs \
       --batch_size $batch_size \
       --lr $lr \
       --weight_decay $weight_decay \
       --num_workers $num_workers \
       --hidden_channels $hidden_channels \
       --num_blocks $num_blocks \
       --solute_model $solute_model \
       --max_confs $max_confs \
       --seed $seed \
       --relative_loss \
       --threshold 50 

echo "End time: $(date '+%Y-%m-%d_%H:%M:%S')"
