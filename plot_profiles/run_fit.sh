#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=cpuq
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH --output=inference.log

source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"
conda activate pyro_env

python fit_diffusion.py \
    --path_to_clipped_profiles  "/scratch/giorgio.ravanelli/test_new_profile/plot_profiles/B207_output_plots/clipped_profiles.pkl" \
    --time 1200 \
    --SLIDEID "SLIDE005" \
    --clip_infletion "True"
