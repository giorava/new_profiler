import os, sys
import argparse
import logging
import re
import numpy as np
import extract_metadata
import subprocess, shlex

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

### retrieve the slide name, the FOV number and the image file name
logging.info(" Reading file names and creating output folders")
list_of_files = os.listdir(args.path_raw_images)
list_of_files = [f for f in list_of_files if os.path.isfile(f"{args.path_raw_images}/{f}")]

for file in list_of_files:
    if file.split(".")[-1] == "nd2": continue
    elif file.split(".")[-1] == "czi": continue
    elif file.split(".")[-1] == "tiff": continue
    elif file.split(".")[-1] == "tif": continue
    else: 
        raise Exception(f"In the raw folder only .nd2, .czi and .tiff files are allowed. {file} not allowed")

regex_slide = "(SLIDE)\d+(?:\d)?"
list_of_slides = np.array([(file_name, 
                            re.search(regex_slide,file_name).group(), file_name.split("_")[-1].split(".")[0], file_name.split("_")[-1].split(".")[1]) for file_name in list_of_files])
unique_slides = np.unique(list_of_slides[:,1])

### Create the output folders
for output_dir in unique_slides: 
    try:
        os.mkdir(f"{args.path_raw_images}/{output_dir}")
    except FileExistsError: 
        raise Exception(f"Output folder {output_dir} already exist! Please avoid overwriting data")

### For each image perform conversion, metadata extraction and deconvolution (to parallelize)
processes = []
threads = args.threads
dw_iterations = args.dw_iterations
for imageID, slideID, fieldID, imageFORM in list_of_slides:

    path_to_image = f"{args.path_raw_images}/{imageID}"
    path_to_output = f"{args.path_raw_images}/{slideID}"

    logging.info(f" Reading metadata {imageID}") 
    obj = extract_metadata.metadata_reader()
    if imageFORM=="czi": 
        ### extract deconvolution metadata
        obj.CZI(path_to_image)
        metadata_dw, n_channels, n_fields = obj.extract_metadata_czi(verbose = False, output = False)

    elif imageFORM=="nd2": 
        ### extract deconvolution metadata
        obj.ND2(path_to_image)
        metadata_dw, n_channels, n_fields = obj.extract_metadata_nd2(verbose = False, output = False) ###

    if n_channels != 2: 
        raise Exception( "Currently implemented only for 2 channel images YFISH and DAPI")

    # metadata commont to all the FOVs
    name_ch0 = metadata_dw['Channel Name (ch 0)'].replace(" ", "")
    name_ch1 = metadata_dw['Channel Name (ch 1)'].replace(" ", "")
    NA, ni = metadata_dw["objective NA"], metadata_dw["ni oil"]
    e_ch0, e_ch1 = metadata_dw["Dye Emission wavelength (nm) (ch 0)"], metadata_dw["Dye Emission wavelength (nm) (ch 1)"]
    dxy, dz = metadata_dw["Pixel size x (nm)"], metadata_dw["Pixel size z (nm)"]

    output_ch0 = f"{path_to_output}/{name_ch0}_{slideID}_{fieldID}.tiff"  
    output_ch1 = f"{path_to_output}/{name_ch1}_{slideID}_{fieldID}.tiff"  

    logging.info(f" Performing conversion {imageID}") 
    with open(f"{path_to_output}/conversion_{slideID}_{fieldID}.log", "w") as f: 
        conv_pipe = [f"bfconvert -channel 0 {path_to_image} '{output_ch0}'", 
                    f"bfconvert -channel 1 {path_to_image} '{output_ch1}'"]
        splitted = [shlex.split(i) for i in conv_pipe]
        conversion_p = [subprocess.Popen(i, stdout=f) for i in splitted]
        for p in conversion_p: p.wait()




logging.info(f" Estimating PSF {imageID}") 
with open(f"{path_to_output}/PSF_{slideID}_{fieldID}.log", "w") as f: 
    dw_PSF_ch0 = shlex.split(f"dw_bw --threads {threads} --NA {NA} --lambda {e_ch0} --ni {ni} --resxy {dxy} --resz {dz} --verbose 2 {path_to_output}/PSF_{NA}_{ni}_{e_ch0}_{dxy}_{dz}.tiff")
    dw_PSF_ch1 = shlex.split(f"dw_bw --threads {threads} --NA {NA} --lambda {e_ch1} --ni {ni} --resxy {dxy} --resz {dz} --verbose 2 {path_to_output}/PSF_{NA}_{ni}_{e_ch1}_{dxy}_{dz}.tiff")
    psf_p = [subprocess.Popen(i, stdout=f) for i in [dw_PSF_ch0, dw_PSF_ch1]]
    for p in psf_p: p.wait()

logging.info(f" Performing deconvolution {imageID}") 
with open(f"{path_to_output}/dw_{slideID}_{fieldID}.log", "w") as f: 
    dw_run_ch0 = shlex.split(f"dw --iter {dw_iterations} --threads {threads} '{output_ch0}' {path_to_output}/PSF_{NA}_{ni}_{e_ch0}_{dxy}_{dz}.tiff")
    dw_run_ch1 = shlex.split(f"dw --iter {dw_iterations} --threads {threads} '{output_ch1}' {path_to_output}/PSF_{NA}_{ni}_{e_ch1}_{dxy}_{dz}.tiff")
    dw_p = [subprocess.run(i, stdout=f) for i in [dw_run_ch0, dw_run_ch1]] 



logging.info(" ---------")
