import os, re, shlex, subprocess 
from scripts.profiler_classes.extract_metadata import *
from scripts.profiler_classes.process_custom_class import *

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
        
def submit_preprocessing(threads, 
                         path_bin,
                         path_raw_folder, 
                         dw_iterations, 
                         perform_decolvolution): 
    
    path_to_raw_images = [f"{path_raw_folder}/"+i for i in os.listdir(path_raw_folder) if (i.split(".")[-1]=="nd2")|(i.split(".")[-1]=="czi")]
    
    for raw_image in path_to_raw_images: 
        print(raw_image)
        obj = ProcessCustom(path_raw_image = raw_image, 
                            dw_iterations = dw_iterations, 
                            threads = threads, 
                            perform_decolvolution = perform_decolvolution, 
                            path_to_bin = path_bin)
        obj.run()
    
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
    
    for folder in [i for i in os.listdir(path_raw_folder) if os.path.isdir(f"{path_raw_folder}/{i}")]: 
        print(folder)
        if re.search("SLIDE", folder): 
            command1 = f"python scripts/plot_fovs.py --image_folder {path_raw_folder}/{folder} --YFISH_channel_name {dapi_channel_name}" 
            command2 = f"python scripts/plot_fovs.py --image_folder {path_raw_folder}/{folder} --YFISH_channel_name {yfish_channel_name}"
            print(command1)
            print(command2)
            splitted = [shlex.split(i) for i in [command1, command2]]
            proc = [subprocess.Popen(i) for i in splitted]
            [i.wait() for i in proc]