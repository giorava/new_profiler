import numpy as np 
import matplotlib.pyplot as plt
import math, os, sys, warnings, re
from scipy.optimize import minimize, Bounds
import pandas as pd
import seaborn
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from scipy import stats

def load_data(paths, name_file): 
    
    dict_data = dict()
    for key in paths: 
        data_frame_path = f"{key}/{name_file}"
        if os.path.isfile(data_frame_path):
            dict_data[key]=pd.read_csv(data_frame_path, sep = "\t")
        else: 
            warnings.warn(f"{data_frame_path} not found")
    
    return dict_data


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


def plot_nuclei_stats(dataset, dataset_key, filtering = False): 
    
    # extract from the dictionary
    nuclei_stats_dapi = dataset[dataset_key]
    
    # computing log10 of area and intensity
    nuclei_stats_dapi["log10_area"] = np.log10(nuclei_stats_dapi["areas"])
    nuclei_stats_dapi["log10_int_intensity"] = np.log10(nuclei_stats_dapi["integrated_intensity"])

    if filtering:
        q10_area, q90_area = np.quantile(nuclei_stats_dapi["areas"], (0.1, 0.90)) 
        small_deb_filt = (nuclei_stats_dapi["areas"]>q10_area)&(nuclei_stats_dapi["areas"]<q90_area)
        nuclei_stats_dapi_filt = nuclei_stats_dapi[small_deb_filt]
        fig = plot_nuc_stats_dist(nuclei_stats_dapi_filt)
    else: 
        fig = plot_nuc_stats_dist(nuclei_stats_dapi)
        
    return fig


def clustering(data, key, plot = True): 
    
    # estimate the clusters
    dataset=data[key]
    dataset["log10_area"] = np.log10(dataset["areas"])
    dataset["log10_int_intensity"] = np.log10(dataset["integrated_intensity"])
    X = dataset[["log10_area", "log10_int_intensity"]]
    y_pred = KMeans(n_clusters=2, n_init = "auto").fit_predict(X)
    dataset["clusters"]=y_pred

    # estimate the center of mass of the clusters
    center_0 = np.array(X[y_pred==0].mean(axis = 0))
    center_1 = np.array(X[y_pred==1].mean(axis = 0))
        
    # do some plotting of the results
    if plot: 
        fig, axs = plt.subplots(1,3, figsize = (20, 5))
        color_dict = {0:"tab:blue", 1:"tab:orange"}
        c_array = [color_dict[i] for i in y_pred]
        
        h = axs[0].scatter(X["log10_area"], X["log10_int_intensity"], c=c_array)
        axs[0].set_xlabel("log10_area")
        axs[0].set_ylabel("log10_int_intensity")
            
        seaborn.kdeplot(data=dataset, x="log10_area", fill=True,
                        ax = axs[1], hue = "clusters")
        seaborn.kdeplot(data=dataset, x="log10_int_intensity", fill=True,
                        ax = axs[2], hue = "clusters")
        
        plt.close()
        return fig, dataset
    else: 
        return dataset
    
    
def find_nuc_ids(data, plot = True): 
    
    lower = {i: np.array([]) for i in data.keys()}
    higher = {i: np.array([]) for i in data.keys()}
    
    for key in data.keys(): 
        df = data[key]
        cl0, cl1 = df[df["clusters"]==0], df[df["clusters"]==1]
        cl0_log10_area_intensity = cl0[["log10_area", "log10_int_intensity"]]
        cl1_log10_area_intensity = cl1[["log10_area", "log10_int_intensity"]]
        
        centroid_cl0 = cl0_log10_area_intensity.mean(axis = 0)
        centroid_cl1 = cl1_log10_area_intensity.mean(axis = 0)
        
        if all(centroid_cl0<centroid_cl1): 
            lower[key] = np.array(cl0["nucleus_label"])
            higher[key] = np.array(cl1["nucleus_label"])
            if plot:
                plt.title(key)
                plt.scatter(cl0_log10_area_intensity["log10_area"], cl0_log10_area_intensity["log10_int_intensity"],
                            color = "tab:blue", alpha = 0.1)
                plt.scatter(cl1_log10_area_intensity["log10_area"], cl1_log10_area_intensity["log10_int_intensity"],
                            color = "tab:orange", alpha = 0.1)
                plt.scatter(*centroid_cl0, label = "lower", color = "black")
                plt.scatter(*centroid_cl1, label = "higher", color = "red")
                plt.xlabel("log10_area")
                plt.ylabel("log10_int_intensity")
                plt.legend()
                plt.show()
            
        elif all(centroid_cl0>centroid_cl1): 
            lower[key] = np.array(cl1["nucleus_label"])
            higher[key] = np.array(cl0["nucleus_label"])
            if plot:
                plt.title(key)
                plt.scatter(cl0_log10_area_intensity["log10_area"], cl0_log10_area_intensity["log10_int_intensity"],
                            color = "tab:blue", alpha = 0.1)
                plt.scatter(cl1_log10_area_intensity["log10_area"], cl1_log10_area_intensity["log10_int_intensity"],
                            color = "tab:orange", alpha = 0.1)
                plt.scatter(*centroid_cl0, label = "higher", color = "black")
                plt.scatter(*centroid_cl1, label = "lower", color = "red")
                plt.xlabel("log10_area")
                plt.ylabel("log10_int_intensity")
                plt.legend()
                plt.show()
            
        else: 
            warnings.warn("Could not assign clusters, use all the data!")
    plt.close()
    return lower, higher


def plot_profiles_boxplot(data_profile_dict, nuc_indexes_dict): 
    
    oredered_keys =  np.array([i for i in data_profile_dict.keys()])
    slide_ids = np.array([float(re.search("SLIDE[0-9][0-9][0-9]",i).group()[-3:]) for i in oredered_keys])
    oredered_keys = oredered_keys[np.argsort(slide_ids)]
    
    for key in oredered_keys: 
        
        profile_df = data_profile_dict[key]
        nuclei_ind = nuc_indexes_dict[key]
        
        selected_profile = profile_df[profile_df["nucleus_label"].isin(nuclei_ind)]
        selected_no_nucid = selected_profile.iloc[:, 1:]
        
        columns = []
        position = []
        for (columnName, columnData) in selected_no_nucid.iteritems(): 
            col_data = np.array(columnData)
            columns.append(col_data[np.isfinite(col_data)])
            position.append(round(float(columnName),2))
            
        labels = []
        for i, pos in enumerate(position): 
            if (i/len(position))%0.2 == 0: 
                labels.append(pos)
            else: 
                labels.append(np.nan)
        
        plt.boxplot(columns, labels = labels)
        plt.title(key)
        plt.show()
        plt.close()


def plot_profiles_polyfit(data_profile_dict, nuc_indexes_dict, degree,
                          output_path, cluster:str): 
    
    oredered_keys =  np.array([i for i in data_profile_dict.keys()])
    slide_ids = np.array([float(re.search("SLIDE[0-9][0-9][0-9]",i).group()[-3:]) for i in oredered_keys])
    oredered_keys = oredered_keys[np.argsort(slide_ids)]
    patter_last_slide = re.search("SLIDE[0-9][0-9][0-9]", oredered_keys[-1]).group()
    patter_slide1 = re.search("SLIDE[0-9][0-9][0-9]", oredered_keys[0]).group()
    
    ## fit profiles
    poly_profiles_unnormed = {i:np.array([]) for i in data_profile_dict.keys()}
    q_profiles_unnormed = {i:np.array([]) for i in data_profile_dict.keys()}
    for key in oredered_keys: 

            profile_df = data_profile_dict[key]
            nuclei_ind = nuc_indexes_dict[key]

            selected_profile = profile_df[profile_df["nucleus_label"].isin(nuclei_ind)]
            selected_no_nucid = selected_profile.iloc[:, 1:]    
            dist = np.array([round(float(i),4) for i in selected_no_nucid.columns])

            ## extract columns
            q20 = np.array(selected_no_nucid.quantile(0.20))
            q50 = np.array(selected_no_nucid.quantile(0.50))
            q80 = np.array(selected_no_nucid.quantile(0.80))

            ## filter quantiles
            d20 = dist[np.isfinite(q20)]
            d50 = dist[np.isfinite(q50)]
            d80 = dist[np.isfinite(q80)]
            q20 = q20[np.isfinite(q20)]
            q50 = q50[np.isfinite(q50)]
            q80 = q80[np.isfinite(q80)]

            ## fit poly
            p20 = np.poly1d(np.polyfit(d20, q20, degree))
            p50 = np.poly1d(np.polyfit(d50, q50, degree))
            p80 = np.poly1d(np.polyfit(d80, q80, degree))

            ## normalize by the max of the first point
            new_x = np.linspace(0, 1, 10000)
            poly_20, poly_50, poly_80 = p20(new_x), p50(new_x), p80(new_x)
            prof = {
                "poly_20":poly_20,
                "poly_50":poly_50,
                "poly_80":poly_80
            }
            qprof = {
                "d_20":d20,
                "d_50":d50,
                "d_80":d80,
                "q_20":q20,
                "q_50":q50,
                "q_80":q80
            }
            q_profiles_unnormed[key] = qprof
            poly_profiles_unnormed[key] = prof

    ## normalize profiles
    norm_factor = [i for i in poly_profiles_unnormed.keys() if re.search(patter_slide1, i)]
    profiles = poly_profiles_unnormed[norm_factor[0]]
    factor = np.max(profiles["poly_50"])

    ## ymax
    norm_factor = [i for i in poly_profiles_unnormed.keys() if re.search(patter_last_slide, i)]
    profiles = poly_profiles_unnormed[norm_factor[0]]
    ymax = np.max(profiles["poly_80"]/factor)*1.1

    ## plot separately
    for idx, key in enumerate(poly_profiles_unnormed.keys()):        
        data_poly = poly_profiles_unnormed[key]
        data_scatter = q_profiles_unnormed[key]

        fig, _ = plt.subplots() 
        x = np.linspace(0,1,10000)
        plt.fill_between(x, data_poly["poly_80"]/factor,
                         data_poly["poly_20"]/factor, color = "gray", alpha = 0.5) 
        plt.scatter(data_scatter["d_50"], data_scatter["q_50"]/factor)
        plt.ylim(-.5, ymax)
        plt.xlabel("Normalize distance from Lamina")
        plt.ylabel("Normalized fluorescence intesnity (a.u.)")
        plt.title(key)
        
        slide_id = re.search("SLIDE[0-9][0-9][0-9]", key).span()
        slide_id = key[slide_id[0]:slide_id[1]]
        fig.savefig(f"{output_path}/{cluster}_{slide_id}_polyfit.png")
        plt.show()
        plt.close()
        
    ## plot together
    ## plot separately
    fig, axs = plt.subplots()
    for idx, key in enumerate(oredered_keys):
        matched_p = re.search("SLIDE[0-9][0-9][0-9]", key).span()
        lab = key[matched_p[0]:matched_p[1]]
        
        data_poly = poly_profiles_unnormed[key]
        data_scatter = q_profiles_unnormed[key]

        # fig, _ = plt.subplots() 
        x = np.linspace(0,1,10000)
        
        plt.fill_between(x, data_poly["poly_80"]/factor,
                         data_poly["poly_20"]/factor, alpha = 0.5, label = lab) 
        plt.plot(data_scatter["d_50"], data_scatter["q_50"]/factor, "o")
        plt.ylim(-.5, ymax)
        plt.xlabel("Normalize distance from Lamina")
        plt.ylabel("Normalized fluorescence intesnity (a.u.)")
        plt.legend()
       
    fig.savefig(f"{output_path}/{cluster}_together_polyfit.png")
    
    plt.close()
    return poly_profiles_unnormed, q_profiles_unnormed




def plot_profiles_polyfit_means(data_profile_dict, nuc_indexes_dict, degree,
                                    output_path, cluster:str): 

    oredered_keys =  np.array([i for i in data_profile_dict.keys()])
    slide_ids = np.array([float(re.search("SLIDE[0-9][0-9][0-9]",i).group()[-3:]) for i in oredered_keys])
    oredered_keys = oredered_keys[np.argsort(slide_ids)]
    patter_last_slide = re.search("SLIDE[0-9][0-9][0-9]", oredered_keys[-1]).group()
    patter_slide1 = re.search("SLIDE[0-9][0-9][0-9]", oredered_keys[0]).group()

    ## fit profiles
    poly_profiles_unnormed = {i:np.array([]) for i in data_profile_dict.keys()}
    q_profiles_unnormed = {i:np.array([]) for i in data_profile_dict.keys()}
    for key in oredered_keys: 

            profile_df = data_profile_dict[key]
            nuclei_ind = nuc_indexes_dict[key]

            selected_profile = profile_df[profile_df["nucleus_label"].isin(nuclei_ind)]
            
            distances_array = []
            mean_array = []
            CI_low_array = []
            CI_high_array = []
            for i in selected_profile.columns: 
                if i!="nucleus_label": 
                    data_column = np.array(selected_profile[i])
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
            
            ### fit with polyfit
            new_dist = np.linspace(0, 1, 1000)
            poly_mean = np.poly1d(np.polyfit(distances_array, mean_array, degree))
            poly_IC_low = np.poly1d(np.polyfit(distances_array, CI_low_array, degree))
            poly_IC_high = np.poly1d(np.polyfit(distances_array, CI_high_array, degree))
            
            poly_profiles_unnormed[key]={
                "distances":new_dist,
                "poly_mean":poly_mean(new_dist),
                "poly_IC_low":poly_IC_low(new_dist),
                "poly_IC_high":poly_IC_high(new_dist)
            }

            q_profiles_unnormed[key]={
                "distances":distances_array,
                "mean":mean_array,
                "IC_low":CI_low_array,
                "IC_high":CI_high_array
            }
            
                
    ## normalize profiles
    norm_factor = [i for i in poly_profiles_unnormed.keys() if re.search(patter_slide1, i)]
    profiles = poly_profiles_unnormed[norm_factor[0]]
    factor = np.max(profiles["poly_mean"])

    ## ymax
    norm_factor = [i for i in poly_profiles_unnormed.keys() if re.search(patter_last_slide, i)]
    profiles = poly_profiles_unnormed[norm_factor[0]]
    ymax = np.max(profiles["poly_IC_high"]/factor)*1.1

    ## plot separately
    for idx, key in enumerate(poly_profiles_unnormed.keys()):        
        data_poly = poly_profiles_unnormed[key]
        data_scatter = q_profiles_unnormed[key]

        fig, _ = plt.subplots() 
        plt.fill_between(data_poly["distances"], data_poly["poly_IC_high"]/factor,
                         data_poly["poly_IC_low"]/factor, color = "gray", alpha = 0.5) 
        plt.scatter(data_scatter["distances"], data_scatter["mean"]/factor)
        plt.ylim(-.5, ymax)
        plt.xlabel("Normalize distance from Lamina")
        plt.ylabel("Normalized fluorescence intesnity (a.u.)")
        plt.title(f"{key} \n mean profile IC = 99%")
        
        slide_id = re.search("SLIDE[0-9][0-9][0-9]", key).span()
        slide_id = key[slide_id[0]:slide_id[1]]
        fig.savefig(f"{output_path}/{cluster}_{slide_id}_polyfit.png")
        plt.show()
        plt.close()
        
    ## plot together
    ## plot separately
    fig, axs = plt.subplots()
    for idx, key in enumerate(oredered_keys):
        matched_p = re.search("SLIDE[0-9][0-9][0-9]", key).span()
        lab = key[matched_p[0]:matched_p[1]]
        
        data_poly = poly_profiles_unnormed[key]
        data_scatter = q_profiles_unnormed[key]

        # fig, _ = plt.subplots() 
        x = np.linspace(0,1,10000)
        
        plt.fill_between(data_poly["distances"], data_poly["poly_IC_high"]/factor,
                         data_poly["poly_IC_low"]/factor, alpha = 0.5, label = lab) 
        plt.plot(data_scatter["distances"], data_scatter["mean"]/factor, "o")
        plt.ylim(-.5, ymax)
        plt.xlabel("Normalize distance from Lamina")
        plt.ylabel("Normalized fluorescence intesnity (a.u.)")
        plt.title("mean profile IC = 99%")
        plt.legend()
       
    fig.savefig(f"{output_path}/{cluster}_together_polyfit.png")
    
    plt.close()
    return poly_profiles_unnormed, q_profiles_unnormed


def plot_profiles_together(poly_profiles_unnormed, q_profiles_unnormed, output_path, cluster):
    
    oredered_keys =  np.array([i for i in poly_profiles_unnormed.keys()])
    slide_ids = np.array([float(re.search("SLIDE[0-9][0-9][0-9]",i).group()[-3:]) for i in oredered_keys])
    oredered_keys = oredered_keys[np.argsort(slide_ids)]
    patter_last_slide = re.search("SLIDE[0-9][0-9][0-9]", oredered_keys[-1]).group()
    patter_slide1 = re.search("SLIDE[0-9][0-9][0-9]", oredered_keys[0]).group()
    

    fig, axs = plt.subplots()
    normalized = {i:np.array([]) for i in poly_profiles_unnormed.keys()}
    for idx, key in enumerate(oredered_keys):
        matched_p = re.search("SLIDE[0-9][0-9][0-9]", key).span()
        lab = key[matched_p[0]:matched_p[1]]
        
        data_poly = poly_profiles_unnormed[key]
        data_scatter = q_profiles_unnormed[key]
        
        factor = np.max(data_poly["poly_50"])

        x = np.linspace(0,1,10000)
        plt.fill_between(x, data_poly["poly_80"]/factor,
                         data_poly["poly_20"]/factor, alpha = 0.5, label = lab) 
        
        plt.plot(data_scatter["d_50"], data_scatter["q_50"]/factor, "o")
        plt.ylim(-.1, 1.5)
        plt.xlabel("Normalize distance from Lamina")
        plt.ylabel("Normalized fluorescence intesnity (a.u.)")
        
        normalized[key] = {
            "x":x,
            "d20":data_scatter["d_20"],
            "d50":data_scatter["d_50"],
            "d80":data_scatter["d_80"],
            "norm_q20":data_scatter["q_20"]/factor,
            "norm_q50":data_scatter["q_20"]/factor,
            "norm_q80":data_scatter["q_20"]/factor,
            "norm_p20":data_poly["poly_20"]/factor,
            "norm_p50":data_poly["poly_50"]/factor,
            "norm_p80":data_poly["poly_80"]/factor
        }
       
    plt.legend()
    fig.savefig(f"{output_path}/{cluster}_together_normed_polyfit.png")
    plt.close()
    return normalized