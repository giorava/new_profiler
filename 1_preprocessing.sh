#!/bin/bash

# sorcing the configuration file
source *.config

echo ">>> submitting PREPROCESSING ${expID} <<<"       
sbatch \
    --partition=cpuq \
    --array=0-$((${number_of_images}-1)) \
    --cpus-per-task=${threads} \
    --mem=${memory_per_image} \
    --job-name="pre${expID}" \
    --time=${preprocessing_estimated_time} \
    --export=path_raw_folder=${path_raw_folder},dw_iterations=${dw_iterations},threads=${threads},path_raw_folder=${path_raw_folder} \
    scripts/preprocessing_cluster.sh

    