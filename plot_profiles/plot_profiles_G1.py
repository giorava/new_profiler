import numpy as np 
import matplotlib.pyplot as plt
import math, os, sys, warnings, re
from scipy.optimize import minimize, Bounds
import pandas as pd
import seaborn
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import pickle
import argparse
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM

from plot_functions import *

def load_data(paths, name_file): 
    
    dict_data = dict()
    for key in paths: 
        data_frame_path = f"{key}/{name_file}"
        if os.path.isfile(data_frame_path):
            dict_data[key]=pd.read_csv(data_frame_path, sep = "\t")
        else: 
            warnings.warn(f"{data_frame_path} not found")
    
    return dict_data

def estimate_CI_mean(data):
    distances_array = []
    mean_array = []
    CI_low_array = []
    CI_high_array = []
    for i in data.columns: 
        if i!="nucleus_label": 
            data_column = np.array(data[i])
            data_filt = data_column[np.isfinite(data_column)]
            if len(data_filt)!=0:
                distance = round(float(i), 4)                    
                samples=[]
                for i in range(1000): 
                    samp = np.random.choice(data_filt, len(data_filt)//4, replace=False)
                    samples.append(samp)                
                
                mean_dist = [np.mean(i) for i in samples]                          
                CI_low = np.mean(mean_dist)-3*np.std(mean_dist)
                CI_high = np.mean(mean_dist)+3*np.std(mean_dist)
            
                distances_array.append(distance)  
                mean_array.append(np.mean(data_filt))
                CI_low_array.append(CI_low)
                CI_high_array.append(CI_high)
    return distances_array, mean_array, CI_low_array, CI_high_array


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Plot the radial profiles')
    parser.add_argument('--expID', type=str,
                            help='Experiment ID')
    parser.add_argument('--dapi_ch_name', type=str,
                            help='Dapi Channel Name')
    parser.add_argument('--yfish_ch_name', type=str,
                            help='YFISH Channel Name')
    args = parser.parse_args()

    ## Set Variables
    pattern=args.expID
    dapi_ch_name=args.dapi_ch_name
    yfish_ch_name=args.yfish_ch_name
    output_plots = f"{args.expID}_output_plots"
    output_nuc_stats = f"{output_plots}/nuclei_selection/selected_nuclei_idexes.pkl"

    ## load datasets
    data_directories=[i for i in os.listdir() if re.search(pattern, i)]
    cy5_profs = load_data(data_directories, f"mean_intensity_profiles_{yfish_ch_name}.tsv")  # load nuclei cy5 mean intensity
    dapi_profs = load_data(data_directories, f"mean_intensity_profiles_{dapi_ch_name}.tsv")  # load nuclei dapi mean intensity
    with open(output_nuc_stats, "rb") as f: 
        nuclei_stats = pickle.load(f)

    
    ## for each key in cy5 and dapi_prof
    # load the nuclei indexes
    # filter the corresponding profile stuff
    # store the remaining profiles
    oredered_keys =  np.array([i for i in nuclei_stats.keys()])
    slide_ids = np.array([float(re.search("SLIDE[0-9][0-9][0-9]",i).group()[-3:]) for i in oredered_keys])
    oredered_keys = oredered_keys[np.argsort(slide_ids)]
    for key in oredered_keys: 
        print(key)
        
        nuclei_ids = nuclei_stats[key]
         
        sele_cy5 = cy5_profs[key]
        sele_dapi = dapi_profs[key]
        sele_cy5 = sele_cy5[sele_cy5["nucleus_label"].isin(nuclei_ids)]
        sele_dapi = sele_dapi[sele_dapi["nucleus_label"].isin(nuclei_ids)]
        
        distances_array, mean_array, CI_low_array, CI_high_array = estimate_CI_mean(sele_cy5)
        
        plt.plot(distances_array, mean_array)
        plt.savefig(key)
        plt.close()