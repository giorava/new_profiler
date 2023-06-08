#!/bin/bash 

# sorcing the configuration file
source *.config

echo ">>> submitting PROFILEs COMPUTATION ${expID} <<<"       
sbatch \
    --partition=cpuq \
    --mem=36GB \
    --job-name="p${expID}" \
    --time="${segmentation_estimated_time}" \
    --export=path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",yfish_channel_name="${yfish_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}" \
    scripts/profiles.sh