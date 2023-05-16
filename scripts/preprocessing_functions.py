import os, sys
import argparse
import logging
import re
import numpy as np
import extract_metadata
import subprocess, shlex


def conversion(path_to_image:str, slideID:str, imageID:str,
               fov_name:str, path_to_output:str,
               output_ch0: str, output_ch1: str) -> None:    

    """
    Conversion of customary images to tiff images using bfconvert
    """ 
    
    logging.info(f" Performing conversion {imageID} fov {fov_name}") 
    with open(f"{path_to_output}/conversion_{slideID}_{fov_name}.log", "w") as f: 
        conv_pipe = [f"bfconvert -channel 0 {path_to_image} '{output_ch0}'",
                        f"bfconvert -channel 1 {path_to_image} '{output_ch1}'"]
        splitted = [shlex.split(i) for i in conv_pipe]
        conversion_p = [subprocess.Popen(i, stdout=f) for i in splitted]
        for p in conversion_p: p.wait()   


def deconvolution(slideID:str, imageID:str,
                  fov_name:str, path_to_output:str,
                  output_ch0: str, output_ch1: str,
                  threads: int, dw_iterations: int,
                  NA: float, e_ch0: float, e_ch1: float,
                  ni: float, dxy: float, dz: float) -> None:     

    """
    PSF estimation and deconvolution of YFISH TIFF images
    """
    
    logging.info(f" Estimating PSF {imageID}") 
    with open(f"{path_to_output}/PSF_{slideID}_{fov_name}.log", "w") as f: 
        dw_PSF_ch0 = shlex.split(f"dw_bw --threads {threads} --NA {NA} --lambda {e_ch0} --ni {ni} --resxy {dxy} --resz {dz} --verbose 2 {path_to_output}/PSF_{NA}_{ni}_{e_ch0}_{dxy}_{dz}.tiff")
        dw_PSF_ch1 = shlex.split(f"dw_bw --threads {threads} --NA {NA} --lambda {e_ch1} --ni {ni} --resxy {dxy} --resz {dz} --verbose 2 {path_to_output}/PSF_{NA}_{ni}_{e_ch1}_{dxy}_{dz}.tiff")
        psf_p = [subprocess.Popen(i, stdout=f) for i in [dw_PSF_ch0, dw_PSF_ch1]]
        for p in psf_p: p.wait()

    logging.info(f" Performing deconvolution {imageID}") 
    with open(f"{path_to_output}/dw_{slideID}_{fov_name}.log", "w") as f: 
        dw_run_ch0 = shlex.split(f"dw --iter {dw_iterations} --threads {threads} '{output_ch0}' {path_to_output}/PSF_{NA}_{ni}_{e_ch0}_{dxy}_{dz}.tiff")
        dw_run_ch1 = shlex.split(f"dw --iter {dw_iterations} --threads {threads} '{output_ch1}' {path_to_output}/PSF_{NA}_{ni}_{e_ch1}_{dxy}_{dz}.tiff")
        dw_p = [subprocess.run(i, stdout=f) for i in [dw_run_ch0, dw_run_ch1]] 

        