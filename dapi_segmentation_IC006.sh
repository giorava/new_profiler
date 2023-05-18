#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --mem=36GB
#SBATCH --job-name=IC006
#SBATCH --output=segmentation_IC006.log

eval "$(conda shell.bash hook)"
conda activate new_profiler

path_to_raw="/scratch/giorgio.ravanelli/test_new_profile/BICRO_IC_006"  ### modify
dapi_channel_name="DAPI"                                                ### modify

# for each folder:
for folder in ${path_to_raw}/*/
do
    python scripts/run_segmentation.py \
        --image_folder ${folder} \
        --dapi_channel_name ${dapi_channel_name}
done
