#!/bin/bash
#SBATCH --job-name=hin
#SBATCH --partition=cpuq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=profiler_hin.log

# load modules and set env_variables
eval "$(conda shell.bash hook)"
conda activate new_profiler

path_to_raw="/scratch/giorgio.ravanelli/test_new_profile/hin" 
dapi_channel_name="dapi" 
yfish_channel_name="A594"
dx_nm="65.0"
dy_nm="65.0"
dz_nm="300.0"

# for each folder:
for folder in ${path_to_raw}/*/
do
    echo ${folder}
    
    python scripts/run_profiles.py \
        --image_folder ${folder} \
        --fluorescence_ch_name ${dapi_channel_name} \
        --pixel_dx ${dx_nm} \
        --pixel_dy ${dy_nm} \
        --pixel_dz ${dz_nm}
    
    python scripts/run_profiles.py \
        --image_folder ${folder} \
        --fluorescence_ch_name ${yfish_channel_name} \
        --pixel_dx ${dx_nm} \
        --pixel_dy ${dy_nm} \
        --pixel_dz ${dz_nm}
        
done
