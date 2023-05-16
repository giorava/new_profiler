#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=cpuq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=segmentation.log

# Load Modules and set env variables
source /home/giorgio.ravanelli/radiantkit_env_v2/bin/activate       ### modify location

path_to_raw="/scratch/giorgio.ravanelli/test_new_profile/raw" 
dapi_channel_name="405nm" 
sigma_gaussian=0.5
gamma_adjust=1

# for each folder:
for folder in ${path_to_raw}/*/
do
    python scripts/run_segmentation.py \
        --image_folder ${folder} \
        --dapi_channel_name ${dapi_channel_name} \
        --sigma_gaussian ${sigma_gaussian} \
        --gamma_adjust ${gamma_adjust}
done