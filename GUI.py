import tkinter as tk
from tkinter import filedialog
import os
import configparser
from preprocessing import *
from segmentation import *
from profile import *

class EnvironmentVariableGUI:
    def __init__(self, root):

        self.config_file = "config.ini"
        self.root = root
        self.root.title("Nuclei Profiler GUI")
        self.path_to_scripts = f"{os.getcwd()}/scripts"
        self.path_to_bin = f"{os.getcwd()}/bin"

        self.standard_frame = tk.LabelFrame(self.root, text="Standard Options")
        self.standard_frame.grid(row=0, columnspan=2, sticky="ew", padx=10, pady=10)  

        self.raw_folder_path_entry = self.create_entry("Raw Folder Path:", 0)
        self.browse_button = tk.Button(self.standard_frame, text="Browse", command=self.browse_directory)
        self.browse_button.grid(row=0, column=2)
        self.browse_button = tk.Button(self.standard_frame, text="Show Metadata", command=self.show_metadata)
        self.browse_button.grid(row=1, column=1)
        
        self.expID_entry  = self.create_entry('Experiment ID', 2)
        self.number_of_images_entry  = self.create_entry('Number of images', 3)
        self.dw_iterations_entry  = self.create_entry('Number of dw iterations', 4)
        self.dapi_channel_name_entry  = self.create_entry('Dapi channel name', 5)
        self.yfish_channel_name_entry  = self.create_entry('YFISH channel name', 6)
        self.pixel_dx_entry = self.create_entry('Pixel dx', 7)
        self.pixel_dy_entry = self.create_entry('Pixel dy', 8)
        self.pixel_dz_entry = self.create_entry('Pixel dz', 9)
        self.estimated_nuc_diameter_entry  = self.create_entry('Estimate nuclei diameter', 10)
        self.user_name_entry = self.create_entry("User name (name.surname)", 22)
        self.squeue_button = tk.Button(self.standard_frame, text="Show queue", command=self.show_queue)
        self.squeue_button.grid(row=22, column=2)
                                                        
        self.advanced_frame = tk.LabelFrame(self.root, text="Advanced Options")
        self.advanced_frame.grid(row=0, column = 2, columnspan=2, sticky="ew", padx=10, pady=10)                                                 
        
        self.preprocessing_estimated_time_entry  = self.show_advance_options('Preprocessing Estimated Time', 11)
        self.preprocessing_estimated_time_entry.insert(0, "03:00:00")
        self.perform_decolvolution_entry  = self.show_advance_options('Perform deconvolution during preprocessing (True/False)', 12)
        self.perform_decolvolution_entry.insert(0, "True")
        self.memory_per_image_entry  = self.show_advance_options('Memory required per image', 13)
        self.memory_per_image_entry.insert(0, "40GB")
        self.threads_entry = self.show_advance_options('Number of Threads', 14)
        self.threads_entry.insert(0, 10)
        self.segmentation_estimated_time_entry = self.show_advance_options('Segmentation Estimated Time', 15)
        self.segmentation_estimated_time_entry.insert(0, "12:00:00")
        self.use_dw_dapi_segmentation_entry = self.show_advance_options('Use deconvolved DAPI for segmentation', 16)
        self.use_dw_dapi_segmentation_entry.insert(0, "True")
        self.standardize_for_segmentation_entry = self.show_advance_options('Standardize image for segmentation', 17)
        self.standardize_for_segmentation_entry.insert(0, "False")
        self.deconvolved_for_profile_entry = self.show_advance_options('Use deconvolved images for profile computation', 18)
        self.deconvolved_for_profile_entry.insert(0, "False")
        
        self.config = tk.LabelFrame(self.root, text="Manage configuration file")
        self.config.grid(row=19, column=0, columnspan=1, sticky="ew", padx=10, pady=10)
        
        self.save_button = tk.Button(self.config, text="Save configuration", command=self.save_to_config)
        self.save_button.grid(row=19, column=3)
        self.delete_button = tk.Button(self.config, text="Clear configuration", command=self.delate_config)
        self.delete_button.grid(row=20, column=3)
        self.export_button = tk.Button(self.config, text="Export configuration", command=self.export_config)
        self.export_button.grid(row=21, column=3)
        
        self.preprocessing_frame = tk.LabelFrame(self.root, text="Preprocessing")
        self.preprocessing_frame.grid(row=19, column=1, columnspan=1, sticky="ew", padx=10, pady=10)
        self.preprocessing_button = tk.Button(self.preprocessing_frame, text="Submit preprocessing", command=self.preprocessing)
        self.preprocessing_button.grid(row=19, column=3)  
        self.preprocessing_button = tk.Button(self.preprocessing_frame, text="Clean folders", command=self.clean_folders_preproc)
        self.preprocessing_button.grid(row=20, column=3)  
        self.preprocessing_button = tk.Button(self.preprocessing_frame, text="Plot FOVS", command=self.plot_fovs)
        self.preprocessing_button.grid(row=21, column=3)  
        
        self.segmentation_frame = tk.LabelFrame(self.root, text="Segmentation")
        self.segmentation_frame.grid(row=19, column=2, columnspan=1, sticky="ew", padx=10, pady=10)
        self.preprocessing_button = tk.Button(self.segmentation_frame, text="Submit segmentation", command=self.segmentation)
        self.preprocessing_button.grid(row=19, column=3)
        
        self.profile_frame = tk.LabelFrame(self.root, text="Compute Profiles")
        self.profile_frame.grid(row=19, column=3, columnspan=1, sticky="ew", padx=10, pady=10)
        self.preprocessing_button = tk.Button(self.profile_frame, text="Submit profiler", command=self.profiler)
        self.preprocessing_button.grid(row=19, column=3) 
        self.preprocessing_button = tk.Button(self.profile_frame, text="After run cleaning", command=self.after_run_cleaning)
        self.preprocessing_button.grid(row=20, column=3) 
        
        self.raw_folder_path = ""
        self.load_config()
        
    def show_advance_options(self, name, row): 
        label = tk.Label(self.advanced_frame, text=name)
        label.grid(row=row, column=0)
        entry= tk.Entry(self.advanced_frame)
        entry.grid(row=row, column=1)
        return entry
            
    def create_entry(self, name, row):
        label = tk.Label(self.standard_frame, text=name)
        label.grid(row=row, column=0)
        entry= tk.Entry(self.standard_frame)
        entry.grid(row=row, column=1)
        return entry        

    def browse_directory(self):
        selected_directory = filedialog.askdirectory()
        if selected_directory:
            self.raw_folder_path = selected_directory
            self.raw_folder_path_entry.delete(0, tk.END)
            self.raw_folder_path_entry.insert(0, self.raw_folder_path)

    def save_to_config(self):
        config = configparser.ConfigParser()
        config['EnvironmentVariables'] = {
            'raw_folder_path': self.raw_folder_path_entry.get(),
            'expID': self.expID_entry.get(),
            'number_of_images': self.number_of_images_entry.get(),
            'dw_iterations': self.dw_iterations_entry.get(),
            'perform_decolvolution': self.perform_decolvolution_entry.get(),
            'dapi_channel_name': self.dapi_channel_name_entry.get(),
            'yfish_channel_name': self.yfish_channel_name_entry.get(),
            'pixel_dx': self.pixel_dx_entry.get(),
            'pixel_dy': self.pixel_dy_entry.get(),
            'pixel_dz': self.pixel_dz_entry.get(),
            'estimated_nuc_diameter': self.estimated_nuc_diameter_entry.get(),
            'preprocessing_estimated_time': self.preprocessing_estimated_time_entry.get(),
            'memory_per_image': self.memory_per_image_entry.get(),
            'threads': self.threads_entry.get(),
            'segmentation_estimated_time': self.segmentation_estimated_time_entry.get(),
            'use_dw_dapi_segmentation': self.use_dw_dapi_segmentation_entry.get(),  
            "user_name": self.user_name_entry.get(), 
            "standardize_image_segmentation": self.standardize_for_segmentation_entry.get(),
            "deconvolved_for_profile":self.deconvolved_for_profile_entry.get()
        }

        with open(self.config_file, 'w') as configfile:
            config.write(configfile)
            
    def delate_config(self): 
        os.remove(self.config_file)   
        
    def export_config(self): 
        os.rename(self.config_file, f"{self.raw_folder_path_entry.get()}/config.ini")
        
    def show_metadata(self): 
        print("\n>>>> Showing metadata")
        display_metadata(self.raw_folder_path_entry.get())
              
    def preprocessing(self): 
        print(f"\n>>>> Submitting preprocessing for {self.expID_entry.get()}")  
        submit_preprocessing(expID = self.expID_entry.get(), 
                            number_of_images = self.number_of_images_entry.get(), 
                            threads = self.threads_entry.get(), 
                            memory_per_image = self.memory_per_image_entry.get(), 
                            preprocessing_estimated_time = self.preprocessing_estimated_time_entry.get(),
                            path_bin = self.path_to_bin, 
                            path_raw_folder = self.raw_folder_path_entry.get(), 
                            dw_iterations = self.dw_iterations_entry.get(), 
                            perform_decolvolution = self.perform_decolvolution_entry.get())
        print(f">>>> Job submitted") 
        
    def show_queue(self): 
        show_queue(self.user_name_entry.get()) 
    
    def profiler(self): 
        print("\n>>>> Submitting profiler")
        run_profile(expID = self.expID_entry.get(), 
                segmentation_estimated_time = self.segmentation_estimated_time_entry.get(), 
                path_raw_folder = self.raw_folder_path_entry.get(), 
                dapi_channel_name = self.dapi_channel_name_entry.get(),
                yfish_channel_name = self.yfish_channel_name_entry.get(), 
                pixel_dx = self.pixel_dx_entry.get(), 
                pixel_dy = self.pixel_dy_entry.get(), 
                pixel_dz = self.pixel_dz_entry.get(), 
                deconvolved_for_profile = self.deconvolved_for_profile_entry.get())
        
        
    def segmentation(self): 
        print("\n>>>> Submitting segmentation") 
        run_segmentation(threads = self.threads_entry.get(),
                     memory_per_image = self.memory_per_image_entry.get(),
                     expID = self.expID_entry.get(), 
                     segmentation_estimated_time = self.segmentation_estimated_time_entry.get(), 
                     path_raw_folder = self.raw_folder_path_entry.get(), 
                     dapi_channel_name = self.dapi_channel_name_entry.get(), 
                     pixel_dx = self.pixel_dx_entry.get(), 
                     pixel_dy = self.pixel_dy_entry.get(), 
                     pixel_dz = self.pixel_dz_entry.get(), 
                     estimated_nuc_diameter = self.estimated_nuc_diameter_entry.get(), 
                     use_dw_dapi = self.use_dw_dapi_segmentation_entry.get(), 
                     standardize_image_for_seg = self.standardize_for_segmentation_entry.get())
        print(f">>>> Job submitted")
        
    def clean_folders_preproc(self): 
        print("\n>>>> Cleaning after preprocessing")  
        clean_folders(self.raw_folder_path_entry.get())
        print(">>>> Finished cleaning after preprocessing")
        
    def plot_fovs(self): 
        print("\n>>>> Plotting FOVs")  
        plot_FOVS(self.raw_folder_path_entry.get(),
                  self.dapi_channel_name_entry.get(),
                  self.yfish_channel_name_entry.get())
        print(">>>> Finished plotting FOVs")
        
    def after_run_cleaning(self): 
        print("\n>>>> After Run cleaning")
        after_run_cleaning(path_raw_folder = self.raw_folder_path_entry.get())
        print("\n>>>> Finished Run cleaning")
                
    def load_config(self):
        if os.path.exists(self.config_file):
            config = configparser.ConfigParser()
            config.read(self.config_file)

            if 'EnvironmentVariables' in config:
                variables = config['EnvironmentVariables']
                if "raw_folder_path" in variables:
                    self.raw_folder_path_entry.delete(0, tk.END)
                    self.raw_folder_path_entry.insert(0, variables['raw_folder_path'])
                if "expID" in variables:
                    self.expID_entry.delete(0, tk.END)
                    self.expID_entry.insert(0, variables['expID'])
                if "number_of_images" in variables:
                    self.number_of_images_entry.delete(0, tk.END)
                    self.number_of_images_entry.insert(0, variables['number_of_images'])
                if "dw_iterations" in variables:
                    self.dw_iterations_entry.delete(0, tk.END)
                    self.dw_iterations_entry.insert(0, variables['dw_iterations'])
                if "perform_decolvolution" in variables:
                    self.perform_decolvolution_entry.delete(0, tk.END)
                    self.perform_decolvolution_entry.insert(0, variables['perform_decolvolution'])
                if "dapi_channel_name" in variables:
                    self.dapi_channel_name_entry.delete(0, tk.END)
                    self.dapi_channel_name_entry.insert(0, variables['dapi_channel_name'])
                if "yfish_channel_name" in variables:
                    self.yfish_channel_name_entry.delete(0, tk.END)
                    self.yfish_channel_name_entry.insert(0, variables['yfish_channel_name'])
                if "pixel_dx" in variables:
                    self.pixel_dx_entry.delete(0, tk.END)
                    self.pixel_dx_entry.insert(0, variables['pixel_dx'])
                if "pixel_dy" in variables:
                    self.pixel_dy_entry.delete(0, tk.END)
                    self.pixel_dy_entry.insert(0, variables['pixel_dy'])
                if "pixel_dz" in variables:
                    self.pixel_dz_entry.delete(0, tk.END)
                    self.pixel_dz_entry.insert(0, variables['pixel_dz'])
                if "estimated_nuc_diameter" in variables:
                    self.estimated_nuc_diameter_entry.delete(0, tk.END)
                    self.estimated_nuc_diameter_entry.insert(0, variables['estimated_nuc_diameter'])
                if "preprocessing_estimated_time" in variables:
                    self.preprocessing_estimated_time_entry.delete(0, tk.END)
                    self.preprocessing_estimated_time_entry.insert(0, variables['preprocessing_estimated_time'])
                if "memory_per_image" in variables:
                    self.memory_per_image_entry.delete(0, tk.END)
                    self.memory_per_image_entry.insert(0, variables['memory_per_image'])
                if "threads" in variables:
                    self.threads_entry.delete(0, tk.END)
                    self.threads_entry.insert(0, variables['threads'])
                if "segmentation_estimated_time" in variables:
                    self.segmentation_estimated_time_entry.delete(0, tk.END)
                    self.segmentation_estimated_time_entry.insert(0, variables['segmentation_estimated_time'])
                if "use_dw_dapi_segmentation" in variables:
                    self.use_dw_dapi_segmentation_entry.delete(0, tk.END)
                    self.use_dw_dapi_segmentation_entry.insert(0, variables['use_dw_dapi_segmentation'])
                if "user_name" in variables: 
                    self.user_name_entry.delete(0, tk.END)
                    self.user_name_entry.insert(0, variables['user_name'])                    
                if "standardize_image_segmentation" in variables: 
                    self.standardize_for_segmentation_entry.delete(0, tk.END)
                    self.standardize_for_segmentation_entry.insert(0, variables['standardize_image_segmentation'])
                if "deconvolved_for_profile" in variables:
                    self.deconvolved_for_profile_entry.delete(0, tk.END)
                    self.deconvolved_for_profile_entry.insert (0, variables['deconvolved_for_profile'])
                
                print("Config loaded successfully.")
        else: 
            print(f"No config present in {self.raw_folder_path_entry.get()}")


# Create the main window
root = tk.Tk()

# Create an instance of the EnvironmentVariableGUI class
app = EnvironmentVariableGUI(root)

# Start the Tkinter event loop
root.mainloop()