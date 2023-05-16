import os, sys
import argparse
import logging
import re
import numpy as np
import extract_metadata
import subprocess, shlex


def conversion(path_to_image:str,
                slideID:str,
                imageID:str,
                fov_name:str,
                path_to_output:str,
                output_ch_list:list[str]) -> None:    

    """
    Conversion of customary images to tiff images using bfconvert
    """ 
    
    logging.info(f" Performing conversion {imageID} fov {fov_name}") 
    with open(f"{path_to_output}/conversion_{slideID}_{fov_name}.log", "w") as f: 

        # create list of commands for conversion of each channel
        conv_pipe = []
        for index, output in enumerate(output_ch_list):
            conv_pipe.append(f"bfconvert -channel {index} {path_to_image} '{output}'")

        # splitting the command and performing the conversion
        splitted = [shlex.split(i) for i in conv_pipe]
        conversion_p = [subprocess.Popen(i, stdout=f) for i in splitted]
        for p in conversion_p: p.wait()   


def deconvolution(slideID:str,
                    imageID:str,
                    fov_name:str, 
                    path_to_output:str,
                    output_ch_list:list[str],
                    threads:int, 
                    dw_iterations:int,
                    NA:float, 
                    ch_lambdas_em:list[str],
                    ni:float, 
                    dxy:float, 
                    dz:float) -> None:  


    """
    PSF estimation and deconvolution of YFISH TIFF images
    """
    
    logging.info(f" Estimating PSF {imageID}") 
    with open(f"{path_to_output}/PSF_{slideID}_{fov_name}.log", "w") as f: 
        
        psf_pipe = []
        for emission, output in zip(ch_lambdas_em, output_ch_list):
            psf_pipe.append(f"dw_bw --threads {threads} --NA {NA} --lambda {emission} --ni {ni} --resxy {dxy} --resz {dz} --verbose 2 {path_to_output}/PSF_{NA}_{ni}_{emission}_{dxy}_{dz}.tiff")
        
        # splitting and running PSF estimation (parallel for each channel)
        splitted = [shlex.split(i) for i in psf_pipe]
        psf_p = [subprocess.Popen(i, stdout=f) for i in splitted]
        for p in psf_p: p.wait()

    logging.info(f" Performing deconvolution {imageID}") 
    with open(f"{path_to_output}/dw_{slideID}_{fov_name}.log", "w") as f: 

        dw_pipe = []
        for emission, output in zip(ch_lambdas_em, output_ch_list): 
            dw_pipe.append(f"dw --iter {dw_iterations} --threads {threads} '{output}' {path_to_output}/PSF_{NA}_{ni}_{emission}_{dxy}_{dz}.tiff")       
        
        # splitting and running deconvolution (in series for each channel)
        splitted = [shlex.split(i) for i in dw_pipe]
        dw_p = [subprocess.run(i, stdout=f) for i in splitted] 
        
