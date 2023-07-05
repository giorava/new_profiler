import os, re, shlex, subprocess 


def run_profile(expID, 
                segmentation_estimated_time, 
                path_raw_folder, 
                dapi_channel_name,
                yfish_channel_name, 
                pixel_dx, 
                pixel_dy, 
                pixel_dz): 
    
    command = f"sbatch --partition=cpuq --mem=36GB --job-name='p{expID}' \
                    --time='{segmentation_estimated_time}' \
                        --output={path_raw_folder}/{expID}_profile.log \
                            --export=expID={expID},path_raw_folder='{path_raw_folder}',dapi_channel_name='{dapi_channel_name}',yfish_channel_name='{yfish_channel_name}',pixel_dx='{pixel_dx}',pixel_dy='{pixel_dy}',pixel_dz='{pixel_dz}' \
                                scripts/profiles.sh"

    print(command)
    splitted = shlex.split(command)
    process = subprocess.Popen(splitted)
    process.wait()    

def after_run_cleaning(path_raw_folder):
    command = f"bash scripts/after_run_cleaning.sh --path_raw_folder {path_raw_folder}"
    print(command)
    splitted = shlex.split(command)
    process = subprocess.Popen(splitted)
    process.wait()    
