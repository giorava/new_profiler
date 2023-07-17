import os, re, shlex, subprocess 
from scripts.profiler_classes.extract_metadata import *
from scripts.profiler_classes.segmentation_dapi import *

def run_segmentation(path_raw_folder, 
                     dapi_channel_name, 
                     pixel_dx, 
                     pixel_dy, 
                     pixel_dz, 
                     estimated_nuc_diameter, 
                     use_dw_dapi, 
                     standardize_image_for_seg): 
    
    
    folders = [i for i in os.listdir(path_raw_folder) if os.path.isdir(f"{path_raw_folder}/{i}")]
    
    for processed_folder in [i for i in folders if re.search("SLIDE", i)]:    
        logging.info(f" PROCESSING {path_raw_folder}")
        seg_obj = DapiSegmentation(
            image_folder = f"{path_raw_folder}/{processed_folder}", 
            dapi_channel_name = dapi_channel_name, 
            dx = float(pixel_dx), dy = float(pixel_dy), dz = float(pixel_dz), 
            nuclei_dimension = int(estimated_nuc_diameter),
            use_dw_dapi = use_dw_dapi,
            standardize_image_for_seg = standardize_image_for_seg
        )
        seg_obj.run_folder()
    