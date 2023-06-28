import scripts.profiler_classes.compute_profiles as compute_profiles
import argparse
import logging
import os

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description = "Run segmentation on dapi images")
    parser.add_argument('--image_folder', type = str,
                        help = "Absolute path to the folder with the preprocessed images")
    parser.add_argument('--fluorescence_ch_name', type = str,
                        help = "Name of the channel to analyise")
    parser.add_argument('--pixel_dx', type = str,
                        help = "pixel dimension in physical space along x axis (nm)")
    parser.add_argument('--pixel_dy', type = str, 
                        help = "pixel dimension in physical space along y axis (nm)")
    parser.add_argument('--pixel_dz', type = str, 
                        help = "pixel dimension in physical space along z axis (nm)")
    args = parser.parse_args()

    # create output folder
    output_folder = f"{args.image_folder}/profiles_output"
    if not(os.path.isdir(output_folder)): 
        os.mkdir(output_folder)

    # instantiate the class and isolate the objects
    pofiler = compute_profiles.ComputeProfiles(
        image_folder=args.image_folder, 
        fluorescence_ch_name=args.fluorescence_ch_name, 
        pixel_dimensions=(float(args.pixel_dz), float(args.pixel_dy), float(args.pixel_dx)) 
    )

    # compute nuclear statistics and save in nuclei_stats.tsv
    nuc_stats = pofiler.nuclear_stats()

    output_stats = f"{output_folder}/nuclei_stats_{args.fluorescence_ch_name}.tsv"
    if not(os.path.exists(output_stats)):
        nuc_stats.to_csv(output_stats, sep = "\t")
    else: 
        logging.info(f" {output_stats} already exists")

    # compute nuclear profiles and save in nuclei_profiles_{args.fluorescence_ch_name}.tsv
    mean_intensity_profiles, median_intensity_profiles = pofiler.nuclear_profiles()
    output_mean = f"{output_folder}/mean_intensity_profiles_{args.fluorescence_ch_name}.tsv"
    output_median = f"{output_folder}/median_intensity_profiles_{args.fluorescence_ch_name}.tsv"
    mean_intensity_profiles.to_csv(output_mean, sep = "\t")
    median_intensity_profiles.to_csv(output_median, sep = "\t")
    