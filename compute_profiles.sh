#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=cpuq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=profiler.log

# load modules and set env_variables
source /home/giorgio.ravanelli/radiantkit_env_v2/bin/activate

path_to_raw="/scratch/giorgio.ravanelli/test_new_profile/raw" 
dapi_channel_name="405nm" 
yfish_channel_name="638nm"
dx_nm="285.0"
dy_nm="285.0"
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
    python scripts/run_profiles.py --image_folder ${folder} --fluorescence_ch_name ${yfish_channel_name} --pixel_dx ${dx_nm} --pixel_dy ${dy_nm} --pixel_dz ${dz_nm}
done
