import os, re, shlex, subprocess 
from scripts.profiler_classes.extract_metadata import *

def display_metadata(path_to_raw: str): 
    
    absolute_paths_to_images = []
    for i in os.listdir(path_to_raw): 
        if re.search(".nd2", i): 
            absolute_paths_to_images.append(f"{path_to_raw}/{i}")
        elif re.search(".czi", i): 
            absolute_paths_to_images.append(f"{path_to_raw}/{i}")

    for path in absolute_paths_to_images:
        basename = os.path.basename(path)
        extension = basename.split(".")[-1]

        if extension == "nd2": 
            obj = metadata_reader()
            obj.ND2(path)
            metadata = obj.extract_metadata_nd2(output = False)

        elif extension == "czi": 
            obj = metadata_reader()
            obj.CZI(path)
            obj.extract_metadata_czi(output = False)

        else: 
            raise Exception('Processing for files different from nd2 and czi is not implemented :S')
        
def submit_preprocessing(expID, 
                         number_of_images, 
                         threads, 
                         memory_per_image, 
                         preprocessing_estimated_time, 
                         path_bin,
                         path_raw_folder, 
                         dw_iterations, 
                         perform_decolvolution): 
    
    number_of_images_array = int(number_of_images)-1
       
    command = f"sbatch --partition=cpuq --array=0-{number_of_images_array} --cpus-per-task={threads} \
                    --mem={memory_per_image} --job-name='pre{expID}' --time={preprocessing_estimated_time} \
                        --output={path_raw_folder}/{expID}_preprocessing_%a.log --export=path_bin={path_bin},path_raw_folder={path_raw_folder},dw_iterations={dw_iterations},threads={threads},perform_decolvolution={perform_decolvolution} \
                            scripts/sbatch_scripts/preprocessing.sh"
    
    print(command)
    
    splitted = shlex.split(command)
    process = subprocess.Popen(splitted)
    process.wait()    
    
def show_queue(user_name): 
    splitted = shlex.split(f"squeue -u {user_name}")
    process = subprocess.Popen(splitted)
    process.wait()    
    
def clean_folders(path_raw_folder): 
    command = f"bash scripts/after_preproc_cleaning.sh --path_raw_folder {path_raw_folder}"
    print(command)
    splitted = shlex.split(command)
    process = subprocess.Popen(splitted)
    process.wait()    

def plot_FOVS(path_raw_folder, dapi_channel_name, yfish_channel_name): 
    command1 = f"bash scripts/plot_fovs.sh --path_raw_folder {path_raw_folder} --channel_name {dapi_channel_name}"
    command2 = f"bash scripts/plot_fovs.sh --path_raw_folder {path_raw_folder} --channel_name {yfish_channel_name}"
    print(command1)
    print(command2)
    
    splitted = [shlex.split(i) for i in [command1, command2]]
    proc = [subprocess.Popen(i) for i in splitted]
    [i.wait() for i in proc]