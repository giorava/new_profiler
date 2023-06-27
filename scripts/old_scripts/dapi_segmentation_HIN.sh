#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --mem=36GB
#SBATCH --job-name=hin
#SBATCH --output=segmentation_hin.log

eval "$(conda shell.bash hook)"
conda activate new_profiler

path_to_raw="/scratch/giorgio.ravanelli/test_new_profile/hin"  ### modify
dapi_channel_name="dapi"                                       ### modify
pixel_dx=65.0                                                  ### modify
pixel_dy=65.0                                                  ### modify
pixel_dz=300.0                                                 ### modify
estimated_nuc_diameter=200                                     ### modify

# for each folder:
for folder in ${path_to_raw}/*/
do
    python scripts/run_segmentation.py \
        --image_folder ${folder} \
        --dapi_channel_name ${dapi_channel_name} \
        --dx ${pixel_dx} --dy ${pixel_dy} --dz ${pixel_dz} \
        --estimated_nuc_diameter ${estimated_nuc_diameter}
done
