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
            if len(data_filt)>1:
                distance = round(float(i), 4)                    
                samples=[]
                for i in range(1000): 
                    samp = np.random.choice(data_filt, len(data_filt)//2, replace=False)
                    samples.append(samp)                
                
                mean_dist = [np.mean(i) for i in samples]                          
                CI_low = np.mean(mean_dist)-3*np.std(mean_dist)
                CI_high = np.mean(mean_dist)+3*np.std(mean_dist)
            
                distances_array.append(distance)  
                mean_array.append(np.mean(data_filt))
                CI_low_array.append(CI_low)
                CI_high_array.append(CI_high)
    
    return np.array(distances_array), np.array(mean_array), np.array(CI_low_array), np.array(CI_high_array)


def plot_profiles(nuclei_stats, dataset_profiles, channel_name): 
    
    oredered_keys =  np.array([i for i in nuclei_stats.keys()])
    slide_ids = np.array([float(re.search("SLIDE[0-9][0-9][0-9]",i).group()[-3:]) for i in oredered_keys])
    oredered_keys = oredered_keys[np.argsort(slide_ids)]
    
    ds = []
    mean_arrays= []
    distance_arrays = []
    mean_fits = []
    CI_low_fits = []
    CI_high_fits = []
    for key in oredered_keys: 
        slide_ID = re.search("SLIDE[0-9][0-9][0-9]",key).group()
        print(f"{channel_name}: {slide_ID}")
        
        nuclei_ids = nuclei_stats[key]
        
        sele = dataset_profiles[key]
        sele = sele[sele["nucleus_label"].isin(nuclei_ids)]
        
        ### cy5 
        distances_array, mean_array, CI_low_array, CI_high_array = estimate_CI_mean(sele)
        
        d, fit_mean = fit_polynomial(distances_array, mean_array, degree = 10, new_x_range = (0,1))
        _, fit_CI_low = fit_polynomial(distances_array, CI_low_array, degree = 10, new_x_range = (0,1))
        _, fit_CI_high = fit_polynomial(distances_array, CI_high_array, degree = 10, new_x_range = (0,1))
        
        ds.append(d)
        distance_arrays.append(distances_array)
        mean_arrays.append(mean_array)
        mean_fits.append(fit_mean)
        CI_low_fits.append(fit_CI_low)
        CI_high_fits.append(fit_CI_high)
        
        fig, axs = plt.subplots()
        plt.fill_between(d, fit_CI_low, fit_CI_high, alpha = 0.5)
        plt.plot(distances_array, mean_array, "o")
        plt.xlabel("Normalized distance from background elements", fontsize = 15)
        plt.ylabel("Fluorescence intensity (a.u.)", fontsize = 15)
        plt.savefig(f"{output_plots_profiles}/{channel_name}_{slide_ID}", bbox_inches='tight')
        plt.close()
        
    ### plot profiles togheter and normalize by the max of the first timepoint
    norm_factor = np.max(mean_fits[0])
    fig, axs = plt.subplots()
    
    for idx, key in enumerate(oredered_keys): 
        slide_ID = re.search("SLIDE[0-9][0-9][0-9]",key).group()
        
        plt.fill_between(ds[idx], CI_low_fits[idx]/norm_factor, CI_high_fits[idx]/norm_factor, alpha = 0.5)
        plt.plot(distance_arrays[idx], mean_arrays[idx]/norm_factor, "o", label = slide_ID)
        
    plt.legend()
    plt.xlabel("Normalized distance from background elements", fontsize = 15)
    plt.ylabel("Fluorescence intensity \n normalized by max of SLIDE001", fontsize = 15)
    plt.savefig(f"{output_plots_profiles}/{channel_name}_together_normalized", bbox_inches='tight')
    plt.close()
    
    return_dict = {
        "ds":ds,
        "mean_arrays": mean_arrays, 
        "distance_arrays": distance_arrays, 
        "mean_fits": mean_fits,
        "CI_low_fits": CI_low_fits,
        "CI_high_fits": CI_high_fits
    }
        
    return return_dict


def fit_polynomial(x_array, y_array, degree, new_x_range):
    
    filtered_y = y_array[np.isfinite(y_array)]
    filtered_x = x_array[np.isfinite(x_array)] 
    polynomial = np.poly1d(np.polyfit(filtered_x, filtered_y, degree))
    new_x = np.linspace(*new_x_range, 1000)
    return new_x, polynomial(new_x)
    


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
    output_plots_profiles = f"{output_plots}/profiles_output"
    
    ## load datasets
    data_directories=[i for i in os.listdir() if re.search(pattern, i)]
    cy5_profs = load_data(data_directories, f"mean_intensity_profiles_{yfish_ch_name}.tsv")  # load nuclei cy5 mean intensity
    dapi_profs = load_data(data_directories, f"mean_intensity_profiles_{dapi_ch_name}.tsv")  # load nuclei dapi mean intensity
    with open(output_nuc_stats, "rb") as f: 
        nuclei_stats = pickle.load(f)

    profiles_dict_yfish = plot_profiles(nuclei_stats = nuclei_stats, dataset_profiles = cy5_profs, channel_name = yfish_ch_name)
    profiles_dict_dapi = plot_profiles(nuclei_stats = nuclei_stats, dataset_profiles = dapi_profs, channel_name = dapi_ch_name)
    
    
    ############## plot YFISH/dapi profiles ##############
    norm_factor = np.max(profiles_dict_yfish["mean_arrays"][0]/profiles_dict_dapi["mean_arrays"][0])
    yfish_over_dapi = []
    for i in range(len(profiles_dict_yfish["mean_arrays"])): 
        name = f"SLIDE00{i+1}"
        signal_over_dapi = profiles_dict_yfish["mean_arrays"][i]/profiles_dict_dapi["mean_arrays"][i]
        yfish_over_dapi.append(signal_over_dapi)
        plt.plot(profiles_dict_yfish["distance_arrays"][i], signal_over_dapi/norm_factor)
        plt.xlabel("Distance form background")
        plt.ylabel("Signal/dapi a.u.")
        plt.savefig(f"{output_plots_profiles}/signal_over_dapi_{name}")
        plt.close()
    
    for i in range(len(profiles_dict_yfish["mean_arrays"])): 
        name = f"SLIDE00{i+1}"
        signal_over_dapi = profiles_dict_yfish["mean_arrays"][i]/profiles_dict_dapi["mean_arrays"][i]
        plt.plot(profiles_dict_yfish["distance_arrays"][i], signal_over_dapi/norm_factor, label = name)
        
    plt.xlabel("Distance form background")
    plt.ylabel("Signal/dapi a.u.")
    plt.legend()
    plt.savefig(f"{output_plots_profiles}/signal_over_dapi_together")
    plt.close()
    #####################################################
    
    
    save_dict = {
        "mean":profiles_dict_yfish["mean_arrays"], 
        "distance":profiles_dict_yfish["distance_arrays"],
        "yfish_over_dapi":yfish_over_dapi
    }
    
    with open(f"{output_plots}/profiles_unormalized.pkl", "wb") as fp: 
        pickle.dump(save_dict, fp) 