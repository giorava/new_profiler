import os, sys
import argparse
import logging
import re
import numpy as np
import extract_metadata
import subprocess, shlex
import warnings
from multiprocessing import *


class ProcessCustom(): 

    def __init__(self, path_raw_image: str, dw_iterations: str, threads: str):
        
        self.path_raw_image = path_raw_image
        self.dw_iterations = dw_iterations
        self.threads = threads

        # find file name and file path
        self.file_name = self.path_raw_image.split("/")[-1]
        self.path_to_raw_folder = "/".join(np.array(self.path_raw_image.split("/"))[:-1])

        # check if file name is nd2 czi or tiff (raise error if not)
        if (self.file_name.split(".")[-1] != "nd2")&(self.file_name.split(".")[-1] != "czi"):
            raise Exception(f"Pre processing implemented only for nd2 and czi files. Check {self.path_raw_image}")

        # check regex and save file name details according to BICRO_HT naming system
        try:
            regex_slide = "(SLIDE)\d+(?:\d)?"
            self.imageID = self.file_name
            self.slideID = re.search(regex_slide, self.file_name).group()
            self.imageFORM = self.file_name.split(".")[-1]
        except AttributeError:
            raise Exception("NAMING for some images in {args.path_raw_images} does not follow agreed convention please check out the SLIDEID")

        # create output directory if necessary
        self.output_folder = f"{self.path_to_raw_folder}/{self.slideID}"
        if not os.path.isdir(self.output_folder): 
            os.mkdir(self.output_folder)       

        # extract deconvolution metadata
        logging.info(f" Reading metadata {self.imageID}") 
        obj = extract_metadata.metadata_reader()
        if self.imageFORM=="czi": 
            obj.CZI(self.path_raw_image)
            self.metadata_dw, self.n_channels, self.n_fields = obj.extract_metadata_czi(verbose = False, output = False)
        elif self.imageFORM=="nd2": 
            obj.ND2(self.path_raw_image)
            self.metadata_dw, self.n_channels, self.n_fields = obj.extract_metadata_nd2(verbose = False, output = False) 
        else: 
            raise Exception(f"Preprocessing implement only for .nd2 and .czi files, please check format of {self.path_raw_image}")
        
        # check the number of channels
        if self.n_channels != 2: 
            warnings.warn("The number of channel does not match the number of channels (YFISH,DAPI) \
                          required for WFISH analysis. Only Preprocessing performed")
            

    def conversion(self, path_to_image:str, slideID:str, imageID:str, fov_name:str,
                   path_to_output:str, output_ch_list:"list[str]", fov_index = None) -> None:    
        
        "Helper function to call conversion from CL with bfconvert"
        
        logging.info(f" Performing conversion {imageID} fov {fov_name}") 
        with open(f"{path_to_output}/conversion_{slideID}_{fov_name}.log", "w") as f: 

            # create list of commands for conversion of each channel
            if fov_index == None: 
                conv_pipe = []
                for index, output in enumerate(output_ch_list):
                    conv_pipe.append(f"bfconvert -channel {index} {path_to_image} '{output}'")
            else: 
                conv_pipe = []
                for index, output in enumerate(output_ch_list):
                    conv_pipe.append(f"bfconvert -series {fov_index} -channel {index} {path_to_image} '{output}'")

            # splitting the command and performing the conversion
            splitted = [shlex.split(i) for i in conv_pipe]
            conversion_p = [subprocess.Popen(i, stdout=f) for i in splitted]
            for p in conversion_p: p.wait()   



    def deconvolution(self, slideID:str, imageID:str, fov_name:str, path_to_output:str, output_ch_list:"list[str]",
                      threads:int, dw_iterations:int, NA:float, ch_lambdas_em:"list[str]", ni:float, dxy:float, 
                      dz:float) -> None: 
        
        "Helper function to call deconvolution from CL with deconwolf"
        
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



    def run(self): 

        if self.n_fields>1: 

            fov_names = [(i, "F"+str(i+1)) for i in range(self.n_fields)]

            for fov_idx, fov_name in fov_names: 

                ch_names = [self.metadata_dw[f'Channel Name (ch {int(i)})'].replace(" ", "") for i in range(self.n_channels)]     # retrieve channel names
                ch_lambdas_em = [self.metadata_dw[f"Dye Emission wavelength (nm) (ch {int(i)})"] for i in range(self.n_channels)] # retrieve channel lambda emissions
                output_ch_list = [f"{self.output_folder}/{name_ch}_{self.slideID}_{fov_name}.tiff" for name_ch in ch_names]      # construct output names
                
                NA, ni = self.metadata_dw["objective NA"], self.metadata_dw["ni oil"]
                dxy, dz = self.metadata_dw["Pixel size x (nm)"], self.metadata_dw["Pixel size z (nm)"]

                kwargs_conversion = {
                    "path_to_image":self.path_raw_image,
                    "slideID":self.slideID,
                    "imageID":self.imageID,
                    "fov_name":fov_name,
                    "path_to_output":self.output_folder,
                    "output_ch_list":output_ch_list, 
                    "fov_index":fov_idx
                }

                kwargs_deconvolution = {                                
                    "slideID":self.slideID,
                    "imageID":self.imageID,
                    "fov_name":fov_name, 
                    "path_to_output":self.output_folder,
                    "output_ch_list":output_ch_list,
                    "threads":self.threads, 
                    "dw_iterations":self.dw_iterations,
                    "NA":NA, 
                    "ch_lambdas_em":ch_lambdas_em,
                    "ni":ni, 
                    "dxy":dxy, 
                    "dz":dz
                }

                self.conversion(**kwargs_conversion)
                self.deconvolution(**kwargs_deconvolution)

                                
        if self.n_fields==1: 

            fov_name = self.imageID.split("_")[-1].split(".")[0]
            if fov_name==self.slideID: 
                raise Exception("file name is not following the agreed guidelines! It should contain a progressive number for the FOV ...<SLIDEID>_<FOVindx>.<format>")

            ch_names = [self.metadata_dw[f'Channel Name (ch {int(i)})'].replace(" ", "") for i in range(self.n_channels)]     # retrieve channel names
            ch_lambdas_em = [self.metadata_dw[f"Dye Emission wavelength (nm) (ch {int(i)})"] for i in range(self.n_channels)] # retrieve channel lambda emissions
            output_ch_list = [f"{self.output_folder}/{name_ch}_{self.slideID}_{fov_name}.tiff" for name_ch in ch_names]      # construct output names
            
            NA, ni = self.metadata_dw["objective NA"], self.metadata_dw["ni oil"]
            dxy, dz = self.metadata_dw["Pixel size x (nm)"], self.metadata_dw["Pixel size z (nm)"]

            kwargs_conversion = {
                    "path_to_image":self.path_to_image,
                    "slideID":self.slideID,
                    "imageID":self.imageID,
                    "fov_name":fov_name,
                    "path_to_output":self.output_folder,
                    "output_ch_list":output_ch_list
                }
            
            kwargs_deconvolution = {                                
                    "slideID":self.slideID,
                    "imageID":self.imageID,
                    "fov_name":fov_name, 
                    "path_to_output":self.output_folder,
                    "output_ch_list":output_ch_list,
                    "threads":self.threads, 
                    "dw_iterations":self.dw_iterations,
                    "NA":NA, 
                    "ch_lambdas_em":ch_lambdas_em,
                    "ni":ni, 
                    "dxy":dxy, 
                    "dz":dz
                }
            
            self.conversion(**kwargs_conversion)
            self.deconvolution(**kwargs_deconvolution)
