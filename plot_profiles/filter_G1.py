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

    
def bootstrapping_and_fit_boxplot(x, y_datas, deg):
        
    samples_mean = []
    samples_low_CI = []
    samples_high_CI = []
    for data in y_datas: 
        samples = []
        for i in range(500):
            samp = np.random.choice(data, len(data)//2, replace=False)
            samples.append(samp)
        _mean = np.mean([np.mean(i) for i in samples])
        _std = np.std([np.mean(i) for i in samples])
        samples_mean.append(_mean)
        samples_low_CI.append(_mean-3*_std)
        samples_high_CI.append(_mean+3*_std)
                    
    pol_fit_mean = np.polyfit(x, samples_mean, deg)
    pol_CI_low = np.polyfit(x, samples_low_CI, deg)
    pol_CI_high = np.polyfit(x, samples_high_CI, deg)
    
    return np.poly1d(pol_fit_mean),  np.poly1d(pol_CI_low),  np.poly1d(pol_CI_high)


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

    ## load datasets
    data_directories=[i for i in os.listdir() if re.search(pattern, i)]
    nuc_stats = load_data(data_directories, f"nuclei_stats_{dapi_ch_name}.tsv")  # load nuclei stats 
    cy5_profs = load_data(data_directories, f"mean_intensity_profiles_{yfish_ch_name}.tsv")  # load nuclei cy5 mean intensity
    dapi_profs = load_data(data_directories, f"mean_intensity_profiles_{dapi_ch_name}.tsv")  # load nuclei dapi mean intensity
    
    ## create plot_directory
    output_plots = f"{args.expID}_output_plots"
    if not os.path.isdir(output_plots):
        os.mkdir(output_plots)
    output_folder = f"{output_plots}/nuclei_selection"
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    output_folder_plots = f"{output_plots}/profiles_output"
    if not os.path.isdir(output_folder_plots):
        os.mkdir(output_folder_plots)
        
    oredered_keys =  np.array([i for i in nuc_stats.keys()])
    slide_ids = np.array([float(re.search("SLIDE[0-9][0-9][0-9]",i).group()[-3:]) for i in oredered_keys])
    oredered_keys = oredered_keys[np.argsort(slide_ids)]

    ## clustering 
    nucl_indexes = {i:np.array([]) for i in oredered_keys}
    stats_list = ""
    areas = []
    for key in oredered_keys: 
        slide_id = re.search("SLIDE[0-9][0-9][0-9]", key).span()
        slide_id = key[slide_id[0]:slide_id[1]]
        
        ### first clustering to get rid of debris
        dataset=nuc_stats[key]
        dataset["log10_area"] = np.log10(dataset["areas"])
        dataset["log10_int_intensity"] = np.log10(dataset["integrated_intensity"])
        X = dataset[["log10_area", "log10_int_intensity"]]
        y_pred = KMeans(n_clusters=2, n_init = "auto").fit_predict(X)
        dataset["cluster"] = y_pred
        
        ### find the higher cluster
        data_cluster_0 = dataset[dataset["cluster"]==0]
        data_cluster_1 = dataset[dataset["cluster"]==1]
        centroid_0 = np.mean(data_cluster_0[["log10_area", "log10_int_intensity"]], axis = 0)
        centroid_1 = np.mean(data_cluster_1[["log10_area", "log10_int_intensity"]], axis = 0)
        
        if all(centroid_0>centroid_1): 
            higher_cluster = data_cluster_0
        elif all(centroid_1>centroid_0): 
            higher_cluster = data_cluster_1
        else: 
            warnings.warn("Cluster location cannot be distinguished uniquely")
        
        ### plot higher cluster 
        high_areas = higher_cluster["areas"]
        high_intensity = higher_cluster["integrated_intensity"]
        x_areas = np.array(high_areas).reshape(-1, 1)
        x_intensity =  np.array(high_intensity).reshape(-1, 1)
        
        ## find the optimal number of components
        bics_area = []
        bics_intensity = []
        min_bic_area = 0
        min_bic_intensity = 0
        opt_bic_area = 0
        opt_bic_intensity = 0
        for i in range(3): 
            gaus_mixture = GMM(n_components = i+1, 
                               max_iter = 1000, 
                               random_state = 0, 
                               covariance_type = 'full')
            labels_areas = gaus_mixture.fit(x_areas).predict(x_areas)
            labels_intensity = gaus_mixture.fit(x_intensity).predict(x_intensity)
            bic_area = gaus_mixture.bic(x_areas)
            bic_intensity = gaus_mixture.bic(x_intensity)
            
            bics_area.append(bic_area)
            bics_intensity.append(bic_intensity)
            
            if bic_area < min_bic_area or min_bic_area == 0:
                if i != 0:
                    min_bic_area = bic_area
                    opt_bic_area = i+1
            if bic_intensity < min_bic_intensity or min_bic_intensity == 0:
                if i != 0:
                    min_bic_intensity = bic_intensity
                    opt_bic_intensity = i+1
                
        ### fit the gaussians
        gmm_area = GMM(n_components = opt_bic_area, 
                        max_iter = 1000, 
                        random_state = 1, 
                        covariance_type = 'full')
        gmm_intensity = GMM(n_components = opt_bic_intensity, 
                        max_iter = 1000, 
                        random_state = 1, 
                        covariance_type = 'full')    
        
        mean_area, cov_area, weights_area = gmm_area.fit(x_areas).means_, gmm_area.fit(x_areas).covariances_, gmm_area.fit(x_areas).weights_
        mean_intensity, cov_intensity, weights_intensity = gmm_intensity.fit(x_intensity).means_, gmm_intensity.fit(x_intensity).covariances_, gmm_intensity.fit(x_intensity).weights_        
        
        area_fits = []
        x_axis_area = np.linspace(0,np.max(x_areas),1000)
        for mean, cov, weight in zip(mean_area, cov_area, weights_area):
            g = norm.pdf(x_axis_area, float(mean[0]), np.sqrt(float(cov[0][0])))*weight
            area_fits.append(g)
        area_fits=np.array(area_fits)
            
        intensities_fits = []
        x_axis_intensity = np.linspace(0,np.max(x_intensity),1000)
        for mean, cov, weight in zip(mean_intensity, cov_intensity, weights_intensity):
            g = norm.pdf(x_axis_intensity, float(mean[0]), np.sqrt(float(cov[0][0])))*weight
            intensities_fits.append(g)     
        intensities_fits=np.array(intensities_fits)   
            
        ### select the G1 gaussian (IF G1 cells are the most abundant)    
        max_gauss_area = np.max(np.max(area_fits, axis = 1))
        max_gauss_intensity = np.max(np.max(intensities_fits, axis = 1))
        gaus_G1_area = area_fits[np.where(np.max(area_fits, axis = 1)==max_gauss_area)[0][0], :]
        gaus_G1_intensity = intensities_fits[np.where(np.max(intensities_fits, axis = 1)==max_gauss_intensity)[0][0], :]
        mean_G1_area = mean_area[np.where(np.max(area_fits, axis = 1)==max_gauss_area)[0][0], :][0]
        mean_G1_intensity = mean_intensity[np.where(np.max(intensities_fits, axis = 1)==max_gauss_intensity)[0][0], :][0]
        std_G1_area = np.sqrt(cov_area[np.where(np.max(area_fits, axis = 1)==max_gauss_area)[0][0], :][0][0])
        std_G1_intensity = np.sqrt(cov_intensity[np.where(np.max(intensities_fits, axis = 1)==max_gauss_intensity)[0][0], :][0][0])
        
        ### G1 interval 
        G1_area_interval = (mean_G1_area-2*std_G1_area, mean_G1_area+2*std_G1_area)
        G1_intensity_interval = (mean_G1_intensity-2*std_G1_intensity, mean_G1_intensity+2*std_G1_intensity)
        
        ### do some plotting
        fig, axs = plt.subplots(1,2, figsize = (15, 5))
        axs[0].vlines(G1_area_interval, 0, np.max(gaus_G1_area))
        axs1 = axs[0].twinx()
        seaborn.kdeplot(high_areas, ax = axs1, fill = True, color = "k")     
        axs[1].vlines(G1_intensity_interval, 0, np.max(gaus_G1_intensity))        
        axs1 = axs[1].twinx()
        seaborn.kdeplot(high_intensity, ax = axs1, fill = True, color = "k")
        axs[1].set_ylim(0, np.max(gaus_G1_intensity)*1.2)
        axs[0].set_ylim(0, np.max(gaus_G1_area)*1.2)
        axs[1].set_title("Integrated intensity", fontsize = 25)
        axs[0].set_title("Area", fontsize = 25)
        plt.savefig(f"{output_folder}/{slide_id}_nuclei_stats")
        
        ### save nuclei indexes of the right cluster
        logical_filter_area = (higher_cluster["areas"]>=G1_area_interval[0])&(higher_cluster["areas"]<=G1_area_interval[1])
        logical_filter_intensity = (higher_cluster["integrated_intensity"]>=G1_intensity_interval[0])&(higher_cluster["integrated_intensity"]<=G1_intensity_interval[1])
        filtered = higher_cluster[logical_filter_intensity&logical_filter_area]            
            
        stats = f"\
        {slide_id}: \n \
            Initial Number of Objects: {dataset.shape[0]}\n \
            After Debried removal: {higher_cluster.shape[0]}\n \
            After G1 filtering: {filtered.shape[0]}\n"
            
            
            ## clustering 
        nucl_indexes[key] = np.array(filtered["nucleus_label"])
        stats_list += stats
    
        areas.append(filtered["areas"])
        print(stats)  
        
        
    fig, axs = plt.subplots()    
    labels = [re.search("SLIDE[0-9][0-9][0-9]",i).group() for i in oredered_keys]
    seaborn.violinplot(areas, color = "tab:orange", axs = axs)
    new_x = np.linspace(0, len(labels)-1)
    _mean, _CI_low, _CI_high =bootstrapping_and_fit_boxplot(np.arange(0, len(labels)), areas, deg = 3)
    plt.fill_between(new_x, _CI_low(new_x), _CI_high(new_x), alpha = 0.5, color = "gray")
    plt.plot(new_x, _mean(new_x), color = "red")
    axs.set_xticks(np.arange(0, len(labels)), labels=labels)
    axs.set_ylabel("Area")
    plt.savefig(f"{output_folder}/areas")
    plt.close()
    
    fig, axs = plt.subplots()    
    labels = [re.search("SLIDE[0-9][0-9][0-9]",i).group() for i in oredered_keys]
    seaborn.violinplot([np.log2(i) for i in areas], color = "tab:orange", axs = axs)    
    _mean, _CI_low, _CI_high =bootstrapping_and_fit_boxplot(np.arange(0, len(labels)), [np.log2(i) for i in areas], deg = 3)
    plt.fill_between(new_x, _CI_low(new_x), _CI_high(new_x), alpha = 0.5, color = "gray")
    plt.plot(new_x, _mean(new_x), color = "red")
    axs.set_xticks(np.arange(0, len(labels)), labels=labels)
    axs.set_ylabel("log10 Area")
    plt.savefig(f"{output_folder}/log10 areas")
    plt.close()
    
    with open(f"{output_folder}/selected_nuclei_idexes.pkl", "wb") as f:
        pickle.dump(nucl_indexes, f)    
            
    with open(f"{output_folder}/stats.txt", 'w') as f:
        f.write(str(stats_list))   
    
        