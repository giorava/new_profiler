import os, re, shlex, subprocess 
from scripts.profiler_classes.compute_profiles import *


def prof(processed_folder, fluorescence_ch_name, pixel_dz, pixel_dy, pixel_dx, deconvolved_for_profile): 
    
    output_folder = f"{processed_folder}/profiles_output"
    if not(os.path.isdir(output_folder)): 
        os.mkdir(output_folder)

    # instantiate the class and isolate the objects
    pofiler = ComputeProfiles(
        image_folder=processed_folder, 
        fluorescence_ch_name=fluorescence_ch_name, 
        pixel_dimensions=(float(pixel_dz), float(pixel_dy), float(pixel_dx)), 
        use_dw = deconvolved_for_profile
    )

    # compute nuclear statistics and save in nuclei_stats.tsv
    nuc_stats = pofiler.nuclear_stats()

    output_stats = f"{output_folder}/nuclei_stats_{fluorescence_ch_name}.tsv"
    if not(os.path.exists(output_stats)):
        nuc_stats.to_csv(output_stats, sep = "\t")
    else: 
        logging.info(f" {output_stats} already exists")

    # compute nuclear profiles and save in nuclei_profiles_{args.fluorescence_ch_name}.tsv
    mean_intensity_profiles, median_intensity_profiles = pofiler.nuclear_profiles()
    output_mean = f"{output_folder}/mean_intensity_profiles_{fluorescence_ch_name}.tsv"
    output_median = f"{output_folder}/median_intensity_profiles_{fluorescence_ch_name}.tsv"
    mean_intensity_profiles.to_csv(output_mean, sep = "\t")
    median_intensity_profiles.to_csv(output_median, sep = "\t")


def run_profile(path_raw_folder, 
                dapi_channel_name,
                yfish_channel_name, 
                pixel_dx, 
                pixel_dy, 
                pixel_dz, 
                deconvolved_for_profile):     

    folders = [f"{path_raw_folder}/{i}" for i in os.listdir(path_raw_folder) if os.path.isdir(f"{path_raw_folder}/{i}")]
    
    for processed_folder in [i for i in folders if re.search("SLIDE", i)]:   
        prof(processed_folder, yfish_channel_name, pixel_dz, pixel_dy, pixel_dx, deconvolved_for_profile)
        prof(processed_folder, dapi_channel_name, pixel_dz, pixel_dy, pixel_dx, deconvolved_for_profile)

def after_run_cleaning(path_raw_folder):
    command = f"bash scripts/after_run_cleaning.sh --path_raw_folder {path_raw_folder}"
    print(command)
    splitted = shlex.split(command)
    process = subprocess.Popen(splitted)
    process.wait()    
