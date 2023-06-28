#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate new_profiler

#declare -a paths_slide_folders=()
for path_raw_image in "${path_raw_folder}"/SLIDE*
do 
    #paths_slide_folders+=( "${path_raw_image}" )
    folder=${path_raw_image}
    echo ">>>> Processing Folder ${folder}"
    echo ">>>> Pixel dim ${pixel_dx}, ${pixel_dy}, ${pixel_dz}"
    echo ">>>> Estimated nuclei pixel dimension ${estimated_nuc_diameter}"
    echo ">>>> Use dw dapi for masks: ${use_dw_dapi}"
    echo ">>>> Standardize image for segmentation: ${standardize_image_for_seg}"

    mkdir -p "${folder}/FOV_plots"
    python scripts/run_segmentation.py \
        --image_folder "${folder}" \
        --dapi_channel_name "${dapi_channel_name}" \
        --dx "${pixel_dx}" --dy "${pixel_dy}" --dz "${pixel_dz}" \
        --estimated_nuc_diameter "${estimated_nuc_diameter}" \
        --use_dw_dapi "${use_dw_dapi}" \
        --standardize_image_for_seg "${standardize_image_for_seg}"
done


