import os, sys
import argparse
import logging
import re
import numpy as np
import extract_metadata
import subprocess, shlex
import preprocessing_functions

logging.basicConfig(level=logging.INFO)

### parse the path to raw images folder
parser = argparse.ArgumentParser(description='Perform all the steps included deconvolution')
parser.add_argument('--path_raw_images', type=str,
                    help='Absolute path to the raw folder')
parser.add_argument('--dw_iterations', type=str,
                    help='Number of iterations')
parser.add_argument('--threads', type=str,
                    help='number of threads')
args = parser.parse_args()

### Retrieve file name, slide index and format of the image
logging.info(" Reading file names and creating output folders")
list_of_files = os.listdir(args.path_raw_images)
list_of_files = [f for f in list_of_files if os.path.isfile(f"{args.path_raw_images}/{f}")]

for file in list_of_files: # check if compatible formats
    if file.split(".")[-1] == "nd2": continue
    elif file.split(".")[-1] == "czi": continue
    elif file.split(".")[-1] == "tiff": continue
    elif file.split(".")[-1] == "tif": continue
    else: 
        raise Exception(f"In the raw folder only .nd2, .czi and .tiff files are allowed. {file} not allowed")

try:
    regex_slide = "(SLIDE)\d+(?:\d)?"
    list_of_slides = np.array([(file_name, 
                                re.search(regex_slide,file_name).group(),
                                file_name.split(".")[-1]) 
                                for file_name in list_of_files])
    unique_slides = np.unique(list_of_slides[:,1])
except AttributeError:
    raise Exception("NAMING for some images in {args.path_raw_images} does not follow agreed convention please check out the SLIDEID")

### Create the output folders
for output_dir in unique_slides: 
    try:
        os.mkdir(f"{args.path_raw_images}/{output_dir}")
    except FileExistsError: 
        raise Exception(f"Output folder {output_dir} already exist! Please avoid overwriting data")

### For each image perform conversion, metadata extraction and deconvolution (to parallelize)
threads = args.threads
dw_iterations = args.dw_iterations

for imageID, slideID, imageFORM in list_of_slides:  

    # save relevant paths
    path_to_image = f"{args.path_raw_images}/{imageID}"
    path_to_output = f"{args.path_raw_images}/{slideID}"

    # read metadata
    logging.info(f" Reading metadata {imageID}") 
    obj = extract_metadata.metadata_reader()
    if imageFORM=="czi": 
        ### extract deconvolution metadata
        obj.CZI(path_to_image)
        metadata_dw, n_channels, n_fields = obj.extract_metadata_czi(verbose = False, output = False)
    elif imageFORM=="nd2": 
        ### extract deconvolution metadata
        obj.ND2(path_to_image)
        metadata_dw, n_channels, n_fields = obj.extract_metadata_nd2(verbose = False, output = False) 
    else: 
        raise Exception(f"Preprocessing implement only for .nd2 and .czi files, please check format of {path_to_image}")

    if n_channels != 2: 
        raise Exception( "Currently implemented only for 2 channel images YFISH and DAPI")

     # convert images to tiff: Multi FOV conversion
    if n_fields>1: 
        fov_names = [(i, "F"+str(i+1)) for i in range(n_fields)]
        name_ch0 = metadata_dw['Channel Name (ch 0)'].replace(" ", "")
        name_ch1 = metadata_dw['Channel Name (ch 1)'].replace(" ", "")
        NA, ni = metadata_dw["objective NA"], metadata_dw["ni oil"]
        e_ch0, e_ch1 = metadata_dw["Dye Emission wavelength (nm) (ch 0)"], metadata_dw["Dye Emission wavelength (nm) (ch 1)"]
        dxy, dz = metadata_dw["Pixel size x (nm)"], metadata_dw["Pixel size z (nm)"]

        for fov_idx, fov_name in fov_names: 
            output_ch0 = f"{path_to_output}/{name_ch0}_{slideID}_{fov_name}.tiff"  
            output_ch1 = f"{path_to_output}/{name_ch1}_{slideID}_{fov_name}.tiff" 
            
            # perform conversion from property format to tiff 
            preprocessing_functions.conversion(
                        path_to_image = path_to_image, slideID = slideID, imageID = imageID,
                        fov_name = fov_name, path_to_output = path_to_output,
                        output_ch0 = output_ch0, output_ch1 = output_ch1)       

            # perform PSF estimation and deconvolution
            preprocessing_functions.deconvolution(
                        slideID = slideID, imageID = imageID,
                        fov_name = fov_name, path_to_output = path_to_output,
                        output_ch0 = output_ch0, output_ch1 = output_ch1,
                        threads = threads, dw_iterations = dw_iterations,
                        NA = NA, e_ch0 = e_ch0, e_ch1 = e_ch1,
                        ni = ni, dxy = dxy, dz = dz)  

    if n_fields==1: 

        fov_name = imageID.split("_")[-1].split(".")[0]
        if fov_name==slideID: 
            raise Exception("file name is not following the agreed guidelines! It should contain a progressive number for the FOV SLIDEID_FOVindx.nd2/czi")

        name_ch0 = metadata_dw['Channel Name (ch 0)'].replace(" ", "")
        name_ch1 = metadata_dw['Channel Name (ch 1)'].replace(" ", "")
        NA, ni = metadata_dw["objective NA"], metadata_dw["ni oil"]
        e_ch0, e_ch1 = metadata_dw["Dye Emission wavelength (nm) (ch 0)"], metadata_dw["Dye Emission wavelength (nm) (ch 1)"]
        dxy, dz = metadata_dw["Pixel size x (nm)"], metadata_dw["Pixel size z (nm)"]

 
        output_ch0 = f"{path_to_output}/{name_ch0}_{slideID}_{fov_name}.tiff"  
        output_ch1 = f"{path_to_output}/{name_ch1}_{slideID}_{fov_name}.tiff"  

        # perform conversion from property format to tiff 
        preprocessing_functions.conversion(
                    path_to_image = path_to_image, slideID = slideID, imageID = imageID,
                    fov_name = fov_name, path_to_output = path_to_output,
                    output_ch0 = output_ch0, output_ch1 = output_ch1)       

        # perform PSF estimation and deconvolution
        preprocessing_functions.deconvolution(
                    slideID = slideID, imageID = imageID,
                    fov_name = fov_name, path_to_output = path_to_output,
                    output_ch0 = output_ch0, output_ch1 = output_ch1,
                    threads = threads, dw_iterations = dw_iterations,
                    NA = NA, e_ch0 = e_ch0, e_ch1 = e_ch1,
                    ni = ni, dxy = dxy, dz = dz)  

logging.info(" ---------")
