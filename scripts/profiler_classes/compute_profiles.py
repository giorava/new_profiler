import numpy as np 
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


class ComputeProfiles(): 
    
    def __init__(self, image_folder: str, fluorescence_ch_name: str, 
                     pixel_dimensions: list, use_dw: str) -> None: 

        self.image_folder = image_folder
        self.mask_folder = f"{image_folder}/masks"
        self.fluorescence_ch_name = fluorescence_ch_name
        self.pixel_dimensions = pixel_dimensions
        self.use_dw = use_dw
        self.file_paths = self.__images_and_masks_paths()
        self.props = self.__compute_properties()
        return None

    def __images_and_masks_paths(self): 

        dw_flo = [f for f in os.listdir(self.image_folder) if re.match(f"dw_{self.fluorescence_ch_name}+.*\.tiff", f)]
        non_dw_flo = [f for f in os.listdir(self.image_folder) if re.match(f"{self.fluorescence_ch_name}+.*\.tiff", f)]

        if self.use_dw=="True": 
            files = [(i, "mask."+i.split("_")[-1].split(".")[0]+".tiff") for i in dw_flo]
            files = [(self.image_folder+"/"+i[0], self.mask_folder+"/"+i[1]) for i in files]
            return np.array(files)

        elif self.use_dw=="False": 
            files = np.array([(i, "mask."+i.split("_")[-1].split(".")[0]+".tiff") for i in non_dw_flo])
            files = [(self.image_folder+"/"+i[0], self.mask_folder+"/"+i[1]) for i in files]
            return np.array(files)
        else: 
            raise Exception(f"No images found in {self.image_folder} for {self.fluorescence_ch_name} with 'Use deconvolved images for profile computation' {self.use_dw}")

    @property
    def get_paths(self): 
        return self.file_paths 
 
    def __compute_properties(self): 
        properties = []
        for flo_path, labels_path in self.file_paths:
            logging.info(f"Computing Properties img {flo_path}")
            labels_img, flo_img = tifffile.imread(labels_path), tifffile.imread(flo_path)
            _prop = measure.regionprops(labels_img, flo_img, 
                                        spacing = self.pixel_dimensions)
            logging.info(f"Found {len(_prop)} objects")
            logging.info(f"------")
            properties += _prop
        return properties

    @property
    def get_prop_objects(self):
        return self.props
    
    def nuclear_stats(self): 

        logging.info(f"Computing nuclei statistics")
        output_dict = {
        "nucleus_label": [i.label for i in self.props], 
        "areas":[i.area for i in self.props],
        "mean_intensity":[np.mean(i.image_intensity) for i in self.props],
        "median_intensity":[np.median(i.image_intensity) for i in self.props],
        "std_intensity":[np.std(i.image_intensity) for i in self.props],
        "integrated_intensity":[np.sum(i.image_intensity) for i in self.props]
        }   
        output_df = pd.DataFrame(output_dict)
        sorted_output = output_df.sort_values("nucleus_label")
        return sorted_output

    def nuclear_profiles(self):

        norm_distance = np.linspace(0, 1, 100)
        mid_bin = np.array((norm_distance[1:]+norm_distance[:-1])/2)

        _dframe = np.zeros((len(self.props),len(mid_bin)))
        nucleus_label = [i.label for i in self.props]

        mean_intensity  = pd.DataFrame(_dframe, index = nucleus_label, columns = mid_bin)
        median_intesity = pd.DataFrame(_dframe, index = nucleus_label, columns = mid_bin)

        logging.info(f" Computing radial profiles for {self.fluorescence_ch_name}")
        for region in tqdm(self.props):
                        
            label = region.label
                
            reg = region.image_intensity
            region_int = np.zeros((reg.shape[0]+2,
                                    reg.shape[1]+2,
                                    reg.shape[2]+2))
            region_int[1:-1, 1:-1, 1:-1] = reg        
                
            # compute pixel distance from background elements
            trasform, idx = ndi.distance_transform_edt(region_int,
                                                        return_indices = True, 
                                                        sampling = self.pixel_dimensions)   
            norm_transform = trasform/np.max(trasform)
            
            # save pixel intensities for each nucleus
            for i in range(1, len(norm_distance)): 
                bin_low  = norm_distance[i-1]
                bin_high = norm_distance[i]
                id_x, id_y, id_z = np.where((norm_transform>bin_low)&(norm_transform<bin_high))

                intensiteis_shell = region_int[id_x, id_y, id_z].copy()
                

                if len(intensiteis_shell)!=0: 
                    median_shell = np.median(intensiteis_shell)
                    mean_shell = np.mean(intensiteis_shell)

                    mean_intensity.loc[label, (bin_low+bin_high)/2] = mean_shell
                    median_intesity.loc[label, (bin_low+bin_high)/2] = median_shell

                else:
                    mean_intensity.loc[label].iloc[i-1] = np.nan
                    median_intesity.loc[label].iloc[i-1] = np.nan
        
        mean_intensity.index.name = "nucleus_label"
        median_intesity.index.name = "nucleus_label"         
        return mean_intensity, median_intesity 
 
    
    