import czifile 
import nd2
import pandas as pd
import tifffile
import numpy as np
import sys,os
import argparse
import logging
logging.basicConfig(level = logging.INFO)


class metadata_reader(): 
    
    def __init__(self): 
        return None
    
    def CZI(self, path_to_czi: str) -> None:
        
        "CZI files init"
        
        logging.info(f" reading: {path_to_czi} ")
        self.img_obj_czi = czifile.czifile.CziFile(path_to_czi)
        
        path_list = path_to_czi.split("/")
        if len(path_list) == 1: 
            self.image_folder_czi = None 
            self.img_name_czi = path_list[0].split(".")[0]
        else: 
            self.image_folder_czi = "/".join(path_list[:-1])
            self.img_name_czi = path_list[-1].split(".")[0]
            
        self.img_array_czi = self.img_obj_czi.asarray()
        self.metadata_czi = self.img_obj_czi.metadata()
        self.metadata_dict_czi = czifile.czifile.xml2dict(self.metadata_czi)["ImageDocument"]["Metadata"]

        num_channels = self.img_array_czi.shape[self.img_obj_czi.axes.index("C")]
        x_dim = self.img_array_czi.shape[self.img_obj_czi.axes.index("X")]
        y_dim = self.img_array_czi.shape[self.img_obj_czi.axes.index("Y")]
        z_dim = self.img_array_czi.shape[self.img_obj_czi.axes.index("Z")]
        self.num_fileds = int(self.img_array_czi.shape[self.img_obj_czi.axes.index("S")])
        image_dimension = f"{x_dim} x {y_dim} x {z_dim}"

        logging.info(f" Number of Channels: {num_channels} ")
        logging.info(f" Number of FOVs: {self.num_fileds}")
        logging.info(f" image_dimension: {image_dimension} ")
        
        if self.img_array_czi.shape[self.img_obj_czi.axes.index("T")] > 1: 
            raise Exception('Conversion for live cell imaging not implemented') 
            
    
    def ND2(self, path_to_nd2: str) -> None: 
        
        "ND2 files init"
        
        logging.info(f" reading: {path_to_nd2} ")
        self.img_obj_nd2 = nd2.ND2File(path_to_nd2)
        
        path_list = path_to_nd2.split("/")
        if len(path_list) == 1: 
            self.image_folder_nd2 = None 
            self.img_name_nd2 = path_list[0].split(".")[0]
        else: 
            self.image_folder_nd2 = "/".join(path_list[:-1])
            self.img_name_nd2 = path_list[-1].split(".")[0]
            
        # num_channels = self.img_obj_nd2.metadata.contents.channelCount
        # x_dim, y_dim, z_dim = self.img_obj_nd2.metadata.channels[0].volume.voxelCount
        # number of fileds in the image
        x_dim, y_dim, z_dim = self.img_obj_nd2.sizes["X"], self.img_obj_nd2.sizes["Y"], self.img_obj_nd2.sizes["Z"]
        
        try: 
            self.num_channels = self.img_obj_nd2.sizes["C"]
        except KeyError: 
            self.num_channels = 1

        try: 
            self.num_fileds = self.img_obj_nd2.sizes["P"]
        except KeyError: 
            self.num_fileds = 1

        self.image_dimension = f"{x_dim} x {y_dim} x {z_dim}"  
        
        logging.info(f" Number of Channels: {self.num_channels} ")
        logging.info(f" Number of FOVs: {self.num_fileds}")
        logging.info(f" image_dimension: {self.image_dimension} ")
            
        self.metadata_nd2 = self.img_obj_nd2.metadata
        
        
    @property
    def get_img_name_czi(self): 
        return self.img_name_czi

    @property
    def get_img_metadata_czi(self): 
        return self.metadata_dict_czi 

    def xmlparser(self, tree_dict: dict, target_attribute: str) -> list: 
        """
        A recursive function to find the attributes in the metadata dictionary
        
            Parameters: 
                tree_dict: the dictionary with the .czi metadata
                target_attribute: the attribute to search for
            Returns: 
                the value of the Target Attribute
        """

        results = []

        for k,v in tree_dict.items(): 
            if k == target_attribute: 
                results += (v,)
            elif isinstance(v, dict): 
                results += self.xmlparser(v, target_attribute)

        return results
    
        
    def save_metadata(self, met_dw: "pd.DataFrame", image_folder: str, image_name: str) -> None: 
        
        if image_folder != None: 
            try:
                os.mkdir(f"{image_folder}/{image_name}")
                met_dw.to_csv(f"{image_folder}/{image_name}/conversion_metadata_{image_name}.log")
                #with open(f"{image_folder}/{image_name}/conversion_metadata_{image_name}.log", "w") as f: 
                #    f.write(met_dw)   
            except FileExistsError: 
                met_dw.to_csv(f"{image_folder}/{image_name}/conversion_metadata_{image_name}.log")
                #with open(f"{image_folder}/{image_name}/conversion_metadata_{image_name}.log", "w") as f: 
                #    f.write(met_dw)    
        else: 
            try: 
                os.mkdir(f"{image_name}")
                met_dw.to_csv(f"{image_name}/conversion_metadata_{image_name}.log")
                #with open(f"{image_name}/conversion_metadata_{image_name}.log", "w") as f: 
                #    f.write(met_dw)   
            except FileExistsError: 
                met_dw.to_csv(f"{image_name}/conversion_metadata_{image_name}.log")
                #with open(f"{image_name}/conversion_metadata_{image_name}.log", "w") as f: 
                #    f.write(met_dw) 

                    
    def extract_metadata_czi(self, verbose: bool = True, output: bool = True) -> None: 
        
        ## save data related to the channels 
        channel_data = []
        for ch_num, ch_info in enumerate(self.xmlparser(self.metadata_dict_czi, "Channel")[0]): 
            _lambda = round(float(ch_info["EmissionWavelength"]))
            try: 
                _ch_name = ch_info["@Name"]
                _ch_id = ch_info["@Id"]
            except: 
                _ch_name = ch_info["Name"]
                _ch_id = ch_info["Id"]

            ch_data = {
                f"Channel Name (ch {ch_num})": _ch_name,
                f"Channel ID (ch {ch_num})": _ch_id,
                f"Dye Emission wavelength (nm) (ch {ch_num})": _lambda
            }

            channel_data.append(pd.Series(ch_data))
            
        #numerical aperture
        objective_NA = round(float(self.xmlparser(self.metadata_dict_czi, "LensNA")[0]), 2)
        
        #pixel sizes 
        dx = round(float(self.xmlparser(self.metadata_dict_czi, "ScalingX")[0])/(1e-9), 2)
        dy = round(float(self.xmlparser(self.metadata_dict_czi, "ScalingY")[0])/(1e-9), 2)
        dz = round(float(self.xmlparser(self.metadata_dict_czi, "ScalingZ")[0])/(1e-9), 2)
        
        #nominal Magnification
        magn = int(self.xmlparser(self.metadata_dict_czi, "NominalMagnification")[0])
        
        #oil refractive index
        oil_refractive_index = float(self.xmlparser(self.metadata_dict_czi, 'RefractiveIndex')[0]) 
        
        # output
        output_ser = pd.Series({
                      "Number of FOVs": self.num_fileds, 
                      "objective NA": objective_NA, 
                      "ni oil": oil_refractive_index, 
                      "Pixel size x (nm)": dx, 
                      "Pixel size y (nm)": dy, 
                      "Pixel size z (nm)": dz,
                      "Nominal Magnification": magn
                     })
        
        met_dw = pd.concat(channel_data+[output_ser])
        
        # Print infos
        if verbose:  
            print(met_dw)   

        if output: 
            self.save_metadata(met_dw, self.image_folder_czi, self.img_name_czi)

        return met_dw, len(channel_data), self.num_fileds

                        
    def extract_metadata_nd2(self, verbose: bool = True, output: bool = True) -> None:
        
        channel_data = []
        for ch_id, ch_metadata in enumerate(self.metadata_nd2.channels): 
            _ch_data = dict()
            _ch_data[f"Channel Name (ch {ch_id})"] = ch_metadata.channel.name
            _ch_data[f"Dye Emission wavelength (nm) (ch {ch_id})"] = ch_metadata.channel.emissionLambdaNm
            
            
            channel_data.append(pd.Series(_ch_data))

        metadata_microscope = dict()
        metadata_microscope["Number of FOVs"] = self.num_fileds
        metadata_microscope["objective NA"] = self.metadata_nd2.channels[0].microscope.objectiveNumericalAperture
        metadata_microscope["ni oil"] = self.metadata_nd2.channels[0].microscope.immersionRefractiveIndex
        pixel_sizes_nm = np.round(np.array(self.metadata_nd2.channels[0].volume.axesCalibration)*1e3)
        metadata_microscope["Pixel size x (nm)"] = pixel_sizes_nm[0]
        metadata_microscope["Pixel size y (nm)"] = pixel_sizes_nm[1]
        metadata_microscope["Pixel size z (nm)"] = pixel_sizes_nm[2]
        metadata_microscope["Nominal Magnification"] = self.metadata_nd2.channels[0].microscope.objectiveMagnification
        
        metadata_microscope = pd.Series(metadata_microscope)
        met_dw = pd.concat(channel_data + [metadata_microscope])
        
        # Print infos
        if verbose:  
            print(met_dw) 

        if output: 
            self.save_metadata(met_dw, self.image_folder_nd2, self.img_name_nd2)

        return met_dw, len(channel_data), self.num_fileds
