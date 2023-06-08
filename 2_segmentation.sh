#!/bin/bash

# sorcing the configuration file
source *.config

# echo ">>> submitting SEGMENTATION ${expID} <<<"       
# sbatch \
#     --partition=gpuq  \
#     --array=0-$((${number_of_images}-1)) \
#     --gres=gpu:2 \
#     --mem-per-gpu=36GB \
#     --job-name="${expID}" \
#     --time="${segmentation_estimated_time}" \
#     --export=path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}",estimated_nuc_diameter="${estimated_nuc_diameter}",use_dw_dapi="${use_dw_dapi}" \
#     scripts/segmentation.sh

# echo ">>> submitting SEGMENTATION ${expID} <<<"       
# sbatch \
#     --partition=cpuq  \
#     --array=0-$((${number_of_images}-1)) \
#     --cpus-per-task=${threads} \
#     --mem=${memory_per_image} \
#     --job-name="${expID}" \
#     --time="${segmentation_estimated_time}" \
#     --export=path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}",estimated_nuc_diameter="${estimated_nuc_diameter}",use_dw_dapi="${use_dw_dapi}" \
#     scripts/segmentation.sh

echo ">>> submitting SEGMENTATION ${expID} <<<"       
sbatch \
    --partition=cpuq  \
    --cpus-per-task=${threads} \
    --mem=${memory_per_image} \
    --job-name="s${expID}" \
    --time="${segmentation_estimated_time}" \
    --export=path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}",estimated_nuc_diameter="${estimated_nuc_diameter}",use_dw_dapi="${use_dw_dapi}" \
    scripts/segmentation.sh