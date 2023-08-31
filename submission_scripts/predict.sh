#!/bin/bash
#SBATCH -J predict_dnpp_s0sp3r
#SBATCH -o ./logs/predictdnpp_xTB_s0sp0sc.log-%j
#SBATCH -e ./logs/predictdnpp_xTB_s0sp0sc.err-%j
##SBATCH -t 1000:00:00
#SBATCH -p xeon-p8
#SBATCH -N 1
#SBATCH --exclusive

source /etc/profile
source activate ConfSolv

which python
echo "\"NODES\"         : \"$(scontrol show hostnames | tr '\n' ',')\","

split_type="scaffold"
split=0
seed=1

CODE=~/Projects/conf_solv
coords_path=$CODE/data/full/dft_coords.pkl.gz
energies_path=$CODE/data/full/free_energies.pkl.gz
split_path=$CODE/data/full/splits/${split_type}_split${split}.pkl
trained_model_dir=$CODE/exps/2023_20_08_23/${split_type}/seed${seed}/egnn

echo "Start time: $(date '+%Y-%m-%d_%H:%M:%S')"
python -u $CODE/conf_solv/predict.py \
       --coords_path $coords_path \
       --energies_path $energies_path \
       --split_path $split_path \
       --trained_model_dir $trained_model_dir \
       --num_workers 0 \
       --n_jobs 1 \
       --no_ionic_solvents

echo "End time: $(date '+%Y-%m-%d_%H:%M:%S')"
