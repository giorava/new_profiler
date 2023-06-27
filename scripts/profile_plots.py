import numpy as np 
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize, Bounds
import pandas as pd
import seaborn
import matplotlib.patches as mpatches
import argparse, os, sys, tqdm


def plot_nuc_stats_dist(nuclei_stats):

    fig, axs = plt.subplots(2,3, figsize = (35, 20))
    axs1, axs2 = axs[0], axs[1]
    
    axs1[1].set_title("Nuclei statistics \n", fontsize = 25)
    seaborn.scatterplot(data=nuclei_stats, x="areas",
                        y="integrated_intensity", ax = axs1[-1])
    seaborn.kdeplot(data=nuclei_stats, x="areas", 
                   fill=True, thresh=0.2, levels=100, ax = axs1[0])
    seaborn.kdeplot(data=nuclei_stats, x="integrated_intensity", 
                   fill=True, thresh=0.2, levels=100, ax = axs1[1])
    
    axs2[1].set_title("Log plot nuclei statistics \n", fontsize = 25)
    seaborn.scatterplot(data=nuclei_stats, x="log10_area",
                        y="log10_int_intensity", ax = axs2[-1])
    seaborn.kdeplot(data=nuclei_stats, x="log10_area", 
                   fill=True, thresh=0.2, levels=100, ax = axs2[0])
    seaborn.kdeplot(data=nuclei_stats, x="log10_int_intensity", 
                   fill=True, thresh=0.2, levels=100, ax = axs2[1])
    
    return fig


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='plot the profiles')
    parser.add_argument('--path_to_processed', type=str,
                        help='Absolute path to the process SLIDEID folder')
    parser.add_argument('--dapi_channel_name', type=str,
                        help='dapi channel name')
    parser.add_argument('--yfish_channel_name', type=str,
                        help='yfish channel name')
    args = parser.parse_args()
    
    output_folders_paths = np.array([(i, f"{args.path_to_processed}/{i}/profiles_output") for i in os.listdir(args.path_to_processed)])
    output_plots_path = f"{args.path_to_processed}/profiles"
    
    try: 
        os.mkdir(output_plots_path)
    except FileExistsError: 
        raise Exception("Output folder already exists :S please double check and remove if empty!")
    
    timepoints = []
    for slideID, output_path in tqdm.tqdm(output_folders_paths): 
        
        # loading data generated from the preprocessing pipeline        
        yfish_intensity_profiles_mean = pd.read_csv(f"{output_path}/mean_intensity_profiles_{args.yfish_channel_name}.tsv", sep = "\t")
        dapi_intensity_profiles_mean = pd.read_csv(f"{output_path}/mean_intensity_profiles_{args.dapi_channel_name}.tsv", sep = "\t")
        nuclei_stats_dapi = pd.read_csv(f"{output_path}/nuclei_stats_{args.dapi_channel_name}.tsv", sep = "\t")
        
        # computing log10 of area and intensity
        nuclei_stats_dapi["log10_area"] = np.log10(nuclei_stats_dapi["areas"])
        nuclei_stats_dapi["log10_int_intensity"] = np.log10(nuclei_stats_dapi["integrated_intensity"])
        
        # filtering for small nuclei and imperfections during the semgentation
        q10_area, q90_area = np.quantile(nuclei_stats_dapi["areas"], (0.1, 0.90)) 
        small_deb_filt = (nuclei_stats_dapi["areas"]>q10_area)&(nuclei_stats_dapi["areas"]<q90_area)
        nuclei_stats_dapi_filt = nuclei_stats_dapi[small_deb_filt]
        
        fig = plot_nuc_stats_dist(nuclei_stats_dapi_filt)
        fig.savefig(f"{output_plots_path}/nuc_stats_{slideID}.png")
        
        
        # nucleus_label	areas	mean_intensity	median_intensity	std_intensity	integrated_intensity        

                
                
        ## selection based on gaussian mixture
        
        
        
        ## plot the profiles on the kept nuclei
        
        
        
        
        
        
        
        
      
      
       
        
    

    

    