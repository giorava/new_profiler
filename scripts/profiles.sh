#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate new_profiler

for path_raw_image in "${path_raw_folder}"/SLIDE*
do 
    folder=${path_raw_image}
    slide_ID=$(basename ${folder})
    echo ">>>> Processing Folder ${folder}" 

    python scripts/run_profiles.py \
    --image_folder ${folder} \
    --fluorescence_ch_name ${dapi_channel_name} \
    --pixel_dx ${pixel_dx} \
    --pixel_dy ${pixel_dy} \
    --pixel_dz ${pixel_dz}
    
    python scripts/run_profiles.py \
        --image_folder ${folder} \
        --fluorescence_ch_name ${yfish_channel_name} \
        --pixel_dx ${pixel_dx} \
        --pixel_dy ${pixel_dy} \
        --pixel_dz ${pixel_dz}

    mv "${path_raw_image}/profiles_output" "${path_raw_image}/${expID}_profiles_output_${slide_ID}"
        
done
