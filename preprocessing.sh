#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=cpuq
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10000
#SBATCH --output=preprocessing.log

# Load Modules and set env variables
module load python/3.9.10
module load fftw/3.3.10-intel-oneapi-mpi-2021.4.0
source /home/giorgio.ravanelli/radiantkit_env_v2/bin/activate       ### modify
PATH=$PATH:/group/bienko/shared/deconwolf-dev/deconwolf-dev/bin    
PATH=$PATH:/group/bienko/shared/TOOLS/bftools

# Raw data location
path_to_raw="/scratch/giorgio.ravanelli/test_new_profile/raw"       ### modify
dw_iterations=15                                                    ### modify 
threads=10                                                          ### modify

# SPLIT THE MULTIFILED images 

python scripts/process_custom.py \
    --path_raw_images ${path_to_raw} \
    --dw_iterations ${dw_iterations} \
    --threads ${threads}

# move all the log files
for output_dir in $(find ${path_to_raw}/* -maxdepth 0 -type d)
do 
    mkdir ${output_dir}/log_files
    mv ${output_dir}/*.log* ${output_dir}/log_files

    mkdir ${output_dir}/PSF
    mv ${output_dir}/PSF_* ${output_dir}/PSF
done
