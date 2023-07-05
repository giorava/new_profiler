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

from plot_functions import *



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

## create plot_directory
output_folder = f"{args.expID}_output_plots"
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
output_folder_lower_cluster = f"{output_folder}/output_folder_lower_cluster"
if not os.path.isdir(output_folder_lower_cluster):
    os.mkdir(output_folder_lower_cluster)
output_folder_higher_cluster = f"{output_folder}/output_folder_higher_cluster"
if not os.path.isdir(output_folder_higher_cluster):
    os.mkdir(output_folder_higher_cluster)

## load datasets
data_directories=[i for i in os.listdir() if re.search(pattern, i)]
nuc_stats = load_data(data_directories, f"nuclei_stats_{dapi_ch_name}.tsv")  # load nuclei stats 
cy5_profs = load_data(data_directories, f"mean_intensity_profiles_{yfish_ch_name}.tsv")  # load nuclei cy5 mean intensity
dapi_profs = load_data(data_directories, f"mean_intensity_profiles_{dapi_ch_name}.tsv")  # load nuclei dapi mean intensity

## plot nuclei stats
clustered_nuc_stats = {i:np.array([]) for i in nuc_stats.keys()}
for key in nuc_stats.keys(): 
    slide_id = re.search("SLIDE[0-9][0-9][0-9]", key).span()
    slide_id = key[slide_id[0]:slide_id[1]]
    fig, data_clusters = clustering(nuc_stats, key, plot = True)
    fig.suptitle(key)
    clustered_nuc_stats[key] = data_clusters
    fig.savefig(f"{output_folder}/nuc_stat_{slide_id}.png")

# divide by the Kmeans clusters    
nuc_idx_lower_cluster, nuc_idx_higher_cluster = find_nuc_ids(clustered_nuc_stats, plot = False)

# plot the profiles fitted with polynomial regression of the HIGHER CLUSTER
poly_profiles_unnormed_yfish, q_profiles_unnormed_yfish = plot_profiles_polyfit(data_profile_dict = cy5_profs,
                                                                    nuc_indexes_dict = nuc_idx_higher_cluster,
                                                                    degree = 10, 
                                                                    output_path = output_folder_higher_cluster, 
                                                                    cluster = "YFISH_higher")
poly_profiles_unnormed_dapi, q_profiles_unnormed_dapi = plot_profiles_polyfit(data_profile_dict = dapi_profs,
                                                                    nuc_indexes_dict = nuc_idx_higher_cluster,
                                                                    degree = 10, 
                                                                    output_path = output_folder_higher_cluster, 
                                                                    cluster = "DAPI_higher")

# plot the profiles with bootstrap to estimate the CI of the mean
poly_mean_IC_yfish, q_mean_IC_yfish = plot_profiles_polyfit_means(data_profile_dict = cy5_profs,
                                                                                        nuc_indexes_dict = nuc_idx_higher_cluster,
                                                                                        degree = 10, 
                                                                                        output_path = output_folder_higher_cluster, 
                                                                                        cluster = "BOOTSTRAP_mean_YFISH_higher")
poly_mean_IC_dapi, q_mean_IC_dapi = plot_profiles_polyfit_means(data_profile_dict = dapi_profs,
                        nuc_indexes_dict = nuc_idx_higher_cluster,
                        degree = 10, 
                        output_path = output_folder_higher_cluster, 
                        cluster = "BOOTSTRAP_mean_DAPI_higher")


normalized = plot_profiles_together(poly_profiles_unnormed_yfish, q_profiles_unnormed_yfish, 
                                    output_path = output_folder_higher_cluster, cluster = "YFISH_higher")
_ = plot_profiles_together(poly_profiles_unnormed_dapi, q_profiles_unnormed_dapi, 
                                    output_path = output_folder_higher_cluster, cluster = "DAPI_higher")


oredered_keys =  np.array([i for i in q_mean_IC_yfish.keys()])
slide_ids = np.array([float(re.search("SLIDE[0-9][0-9][0-9]",i).group()[-3:]) for i in oredered_keys])
oredered_keys = oredered_keys[np.argsort(slide_ids)]

fig, axs = plt.subplots()
clipped_prof = {i:np.array([]) for i in q_mean_IC_yfish.keys()}
clipped_prof_norm = {i:np.array([]) for i in q_mean_IC_yfish.keys()}
for key in oredered_keys: 
    slide_id = re.search("SLIDE[0-9][0-9][0-9]", key).span()
    slide_id = key[slide_id[0]:slide_id[1]]
    
    data_points = q_mean_IC_yfish[key]
    poly_points = poly_mean_IC_yfish[key]
    
    dist_points, mean_points = data_points["distances"], data_points["mean"]    
    dist_poly, mean_poly = poly_points["distances"], poly_points["poly_mean"] 
    
    norm_mean_poly = mean_poly/np.max(mean_poly)
    where_1 = np.where(mean_poly==np.max(mean_poly))[0]
    x_position_max = dist_poly[where_1]
    print(f"Max location {slide_id} : {x_position_max[0]}")
    
    where_d = np.where(dist_points>x_position_max)[0][0]
    distance_filtered, mean_filtered = dist_points[where_d:], mean_points[where_d:]
    clipped_d = np.linspace(0, 1, len(distance_filtered))
    
    clipped_prof[key] = {
        "d_after_max":distance_filtered,
        "d_clipped":clipped_d, 
        "points_prof_clipped":mean_filtered, 
        "normalized_points":mean_filtered/np.max(mean_filtered)
    }
    
    plt.plot(clipped_d, mean_filtered, "o", label = slide_id)
    plt.xlabel("Clipped distance to the max")
    plt.ylabel("Mean normalized YFISH signal")
    plt.xlim(-0.1,1.1)
        
plt.legend()
fig.savefig(f"{output_folder}/clipped.png")
plt.close()


# save dictionary to clipped_profiles.pkl file
with open(f'{output_folder}/clipped_profiles.pkl', 'wb') as fp:
    pickle.dump(clipped_prof, fp)
    
    
    
# plot the profiles fitted with polynomial regression of the LOWER CLUSTER
poly_profiles_unnormed_yfish, q_profiles_unnormed_yfish = plot_profiles_polyfit(data_profile_dict = cy5_profs,
                                                                    nuc_indexes_dict = nuc_idx_lower_cluster,
                                                                    degree = 10, 
                                                                    output_path = output_folder_lower_cluster, 
                                                                    cluster = "YFISH_lower")
poly_profiles_unnormed_dapi, q_profiles_unnormed_dapi = plot_profiles_polyfit(data_profile_dict = dapi_profs,
                                                                    nuc_indexes_dict = nuc_idx_lower_cluster,
                                                                    degree = 10, 
                                                                    output_path = output_folder_lower_cluster, 
                                                                    cluster = "DAPI_lower")

try:
    plot_profiles_polyfit_means(data_profile_dict = cy5_profs,
                            nuc_indexes_dict = nuc_idx_lower_cluster,
                            degree = 10, 
                            output_path = output_folder_lower_cluster, 
                            cluster = "BOOTSTRAP_mean_YFISH_lower")
    plot_profiles_polyfit_means(data_profile_dict = dapi_profs,
                            nuc_indexes_dict = nuc_idx_lower_cluster,
                            degree = 10, 
                            output_path = output_folder_lower_cluster, 
                            cluster = "BOOTSTRAP_mean_DAPI_lower")
except:
    print("Not plotting IC for lower cluster")

normalized = plot_profiles_together(poly_profiles_unnormed_yfish, q_profiles_unnormed_yfish, 
                                    output_path = output_folder_lower_cluster, cluster = "YFISH_lower")
_ = plot_profiles_together(poly_profiles_unnormed_dapi, q_profiles_unnormed_dapi, 
                                    output_path = output_folder_lower_cluster, cluster = "DAPI_lower")

