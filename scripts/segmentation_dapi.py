import numpy as np 
from cellpose import models
import skimage
import scipy
import pandas as pd
import math
import tifffile
import os
import cv2
import re
from skimage.filters import sobel
from skimage.segmentation import watershed, clear_border
from scipy import ndimage as ndi
from tqdm import tqdm
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit
import logging
logging.basicConfig(level=logging.INFO)


class DapiSegmentation(): 

    def __init__(self, image_folder: str, dapi_channel_name: str):

        self.image_folder = image_folder
        self.dapi_ch_name = dapi_channel_name
        return None


    def find_FOVs_list(self): 
        
        # select only dapi channel 
        #dapi_tiffs = [f for f in os.listdir(self.image_folder) if re.match(f"{self.dapi_ch_name}+.*\.tiff", f)]
        dapi_tiffs = [f for f in os.listdir(self.image_folder) if re.match(f"dw_{self.dapi_ch_name}+.*\.tiff", f)]
        
        # extract the FOV idx (must be the last one)
        fov_list = np.array([(i, i.split("_")[-1].split(".")[0]) for i in dapi_tiffs])

        # order the list
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        sorted_FOVS = sorted(fov_list[:, 1], key = alphanum_key)
        sorted_tiffs = np.array([fov_list[fov_list[:,1]==i][0][0] for i in sorted_FOVS])

        return sorted_tiffs.flatten()

    def clean_xy_borders(self, labels_image): 
    
        labels_to_delate = np.array([])
        for z in range(labels_image.shape[0]): 

            plane = labels_image[z, :, :].copy()
            plane[1:-1, 1:-1] = 0
            uniq_labs = np.unique(plane[plane!=0])
            labels_to_delate = np.concatenate([labels_to_delate, uniq_labs])    
             
        to_remove = np.unique(labels_to_delate)
        new_labels = labels_image.copy()
        
        for element in to_remove: 
            new_labels[new_labels==element] = 0

        return new_labels

    def segmentation_single_image(self, dapi_tiff_image: str) -> np.ndarray: 

        # load dapi image and convert in array
        logging.info(f" ---------")
        logging.info(f" Loading {dapi_tiff_image}")
        image_dapi_obj = tifffile.tifffile.TiffFile(dapi_tiff_image)
        stack_image_dapi = image_dapi_obj.asarray()
        image_dapi_obj.close()

        ##################################################################  
        ####### Performing segmentation with CellPose nuclei NN ##########
        ##################################################################

        model = models.Cellpose(gpu=True, model_type='nuclei')

        logging.info(f"Starting Mask file generation")
        labels, flows, styles, diams = model.eval(stack_image_dapi, 
                                                  diameter=None,
                                                  channels=[0,0],
                                                  do_3D=True)

        ##################################################################

        logging.info(" Clear borders") 
        clean_labels = self.clean_xy_borders(labels)

        return  clean_labels


    def run_folder(self) -> None: 

        # identify the dapi tiff images based on self.dapi_ch_name
        sorted_fileds = self.find_FOVs_list()

        # create self.image_folder/masks
        try:
            os.mkdir(f"{self.image_folder}/masks")
        except FileExistsError: 
            raise Exception(f"{self.image_folder}/masks already exist! Please avoid overwriting data")

        number_of_nuclei = 0
        nuclei_count = []
        for dapi_file in sorted_fileds: 
            image_path = f"{self.image_folder}/{dapi_file}"
            labels_FOV = self.segmentation_single_image(image_path)
            nuclei_labels = np.unique(labels_FOV[labels_FOV!=0])

            labels_FOV_upd = labels_FOV.copy()
            labels_FOV_upd[labels_FOV_upd!=0] += number_of_nuclei

            number_of_nuclei += len(nuclei_labels)
            nuclei_count.append(len(nuclei_labels))

            otput_mask_name = f"mask.{dapi_file.split('_')[-1]}"
            tifffile.imwrite(f"{self.image_folder}/masks/{otput_mask_name}", labels_FOV_upd)

