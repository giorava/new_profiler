import os, re, shlex, subprocess 
from scripts.profiler_classes.extract_metadata import *

def run_segmentation(threads,
                     memory_per_image,
                     expID, 
                     segmentation_estimated_time, 
                     path_raw_folder, 
                     dapi_channel_name, 
                     pixel_dx, 
                     pixel_dy, 
                     pixel_dz, 
                     estimated_nuc_diameter, 
                     use_dw_dapi, 
                     standardize_image_for_seg): 
    
    command = f"sbatch --partition=gpuq --gres=gpu:1 --cpus-per-gpu={threads} --mem={memory_per_image} \
    --job-name='s{expID}' --time='{segmentation_estimated_time}' --output={path_raw_folder}/{expID}_segmentation_GPU.log \
    --export=path_raw_folder='{path_raw_folder}',dapi_channel_name='{dapi_channel_name}',pixel_dx='{pixel_dx}',pixel_dy='{pixel_dy}',pixel_dz='{pixel_dz}',estimated_nuc_diameter='{estimated_nuc_diameter}',use_dw_dapi='{use_dw_dapi}',standardize_image_for_seg='{standardize_image_for_seg}' \
    scripts/segmentation.sh"
    print(command)
    splitted = shlex.split(command)
    process = subprocess.Popen(splitted)
    process.wait()    