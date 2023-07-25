# New profiler (RadiantKit2.0)

The general purpose of the application is to detect and quantify the radial distribution of fluorescent signals in nuclei imaged with a variety of wide-field fluorescent microscopy techniques (see https://www.nature.com/articles/s41587-020-0519-y). To do so, we developed a python3-based analysis pipeline thought to run remotely on the Human Technopole HPC cluster together with a user-friendly GUI. 

## General workflow of the analysis
The general workflow of the pipeline is exemplified in the figure. Briefly: 
1.	**Pre-processing**: proprietary format images (.czi or .nd2) are converted to .tiff format and split into multiple Fields of View and channels (pipeline implemented only for two channels images). Each channel is deconvolved using software previously developed by the Bienko lab. 
2.	**Mask generation**: the dapi channel generated in the previous step is then segmented with CellPose (Stinger et al. Nat methods 2021) to retrieve the mask files for the nuclei. 
3.	**Profile generation**: the final radial profiles are then computed using the exact Euclidean distance from background elements determined on the mask files of step 2. 
The output of the analysis is a series of .tsv files containing the profile output for each nucleus and statistics about the nuclei's shape, area etc. Currently, the profiles are plotted using separate scripts in plot_profiles. 



<br>
<figure>  
<p align="center">
<img src="https://github.com/giorava/new_profiler/assets/107054086/7a6f2655-80c6-40fc-a66d-605bd1f59949" width="900"></a>
<figcaption>  
  <p align="center">
    Figure 1. General workflow of the RadiantKit analysis. 
  </p>
</figcaption>
</p>
</figure>

## Installation
To install the package it is enough to clone this repository and install the dependencies (assuming conda is installed in your HT environment): 
```
conda env create -f profiler.yml
```
If you are not using the HT cluster please make sure that the binaries in /bin are working: 
```
${path_to_bin}/dw --help
${path_to_bin}/dw_bw --help
${path_to_bin}/bftools/bfconvert --help
```
you should see the help page of each dependency.


## Running with the GUI

The package contains a graphical interphase that can be used to run all the steps. To access GUI.py while keeping all the analysis running on the cluster you need to install MobaXterm (if Windows user) or xquartz (if Mac user). 
To run the analysis using the GUI follow these steps: 

1. log into the cluster using X11 forwarding:
    ```
    ssh -YC ${HT_USER}@${loginNode}
    ```
2. start an interactive session with X11 forwarding:
    ```
    srun --nodes=1 --tasks-per-node=1 --x11 --mem=30GB --partition=cpu-interactive --pty /bin/bash
    ```
3. activate the profiler environment:
    ```
    conda activate new_profiler
    ```
4. run the GUI
    ```
    cd ${path_to_repo}
    python GUI.py
    ```

The following window will appear: 

<br>
<figure>  
<p align="center">
<img src="https://github.com/giorava/new_profiler/assets/107054086/9795abaf-d899-41c2-bb28-94aecf71aa36" width="900"></a>
<figcaption>  
  <p align="center">
    GUI window.
  </p>
</figcaption>
</p>
</figure>



<br>
<h3>
  To run the <strong>Preprocessing</strong>:
</h3>

<br>
<figure>  
<p align="center">
<img src="https://github.com/giorava/new_profiler/assets/107054086/05f2cd30-c5e1-4929-885a-017c6b0e27a3" width="600"></a>
</p>
</figure>


1. Select the raw data folder using the `Browse` button.
2. Display the metadata with `Show Metadata` and use this info to fill in the standard option fields.
3. Fill in your account details in the `User Name` field.
4. Double-check the advanced options. Particularly, `Preprocessing Estimated Time`, `Perform Deconvolution during preprocessing` and `Memory required per image`.
5. Press `Submit preprocessing`.


At this stage, the preprocessing is running. You can monitor the job using the `Show queue` button. Once the preprocessing is finished you can proceed with `Clean folders` and `Plot FOVS`. 

<br>
<h3>
  To run the <strong>Segmentation</strong>:
</h3>

<br>
<figure>  
<p align="center">
<img src="https://github.com/giorava/new_profiler/assets/107054086/034690a2-6265-4bd7-95ac-82834e41f321" width="600"></a>
</p>
</figure>

1. Adjust the `Estimated nuclei diameter` using the FOV plots of the preprocessing step, or by opening the images in Fiji.
2. Double check the advanced fields `Segmention Estimated Time`, `Use deconvolved DAPI for segmentation` and `Standardize image for segmentation`.
3. Press `Submit segmentation`

> **Warning**
> With FOVs with very few cells the segmentation might fail with the default options. You might see some over-splitting. In this case, try:
> 
> 1. Increasing the nuclei diameter
> 2. Setting `Standardize image for segmentation` to True
> 3. Setting use deconvolved DAPI to False
>    
> Please try only one of these options at the time. If the problem persists consider using an alternative segmentation procedure (to be implemented soon) or retrain CellPose on your data.

<br>
<h3>
To run the <strong>Profile Computation</strong>:
</h3>

<br>
<figure>  
<p align="center">
<img src="https://github.com/giorava/new_profiler/assets/107054086/048853ec-469b-4668-8d25-4a52f98a7806" width="600"></a>
</p>
</figure>


1. Double-check the mask files.
2. Double check the advanced field `Use deconvolved images for profile computation`
3. Press `Submit profiler`

Once the job is completed, you can reorganize the output folder by using `After run cleaning`


<br>
<h2>
Output structure
</h2>

Before the preprocessing, the raw folder should contain only the raw images. It is ok if the file is multi-FOV. The only strick requirement is that the file name should contain SLIDE### identifying the ID of the corresponding slide.
```
└── test_SLIDE001_01.nd2
```
After the preprocessing, cleaning and FOVs plotting a new folder named following the SLIDE### is created. In this folder the tiff files of the splitted channels and the corresponding deconvolved images are present. There are three additional subfolders: FOV_plots, containing png images of the middle plane of each field of view; log_files, containing the log files of conversion and deconvolution; PSF containing the point spread function models used for the deconvolution. 
```
├── SLIDE001                                                          # output folder
│   ├── CY5_SLIDE001_01.tiff
│   ├── DAPI_SLIDE001_01.tiff
│   ├── dw_CY5_SLIDE001_01.tiff
│   ├── dw_DAPI_SLIDE001_01.tiff
│   ├── FOV_plots
│   │   ├── dw_CY5_SLIDE001_01.png
│   │   └── dw_DAPI_SLIDE001_01.png
│   ├── log_files
│   │   ├── conversion_SLIDE001_01.log
│   │   ├── dw_CY5_SLIDE001_01.tiff.log.txt
│   │   ├── dw_DAPI_SLIDE001_01.tiff.log.txt
│   │   ├── dw_SLIDE001_01.log
│   │   ├── PSF_1.25_1.406_438.0_284.0_300.0.tiff.log.txt
│   │   ├── PSF_1.25_1.406_684.0_284.0_300.0.tiff.log.txt
│   │   └── PSF_SLIDE001_01.log
│   └── PSF
│       ├── PSF_1.25_1.406_438.0_284.0_300.0.tiff
│       └── PSF_1.25_1.406_684.0_284.0_300.0.tiff
├── TEST_preprocessing_0.log
└── test_SLIDE001_01.nd2
```

After the segmentation, an additional folder is created containing the mask files of the FOV. During the mask generation, the corresponding middle plane images are stored as png in the FOV_plots folder for subsequent inspection. Please double-check the goodness of the mask generation before proceeding further.
```
├── SLIDE001
│   ├── CY5_SLIDE001_01.tiff
│   ├── DAPI_SLIDE001_01.tiff
│   ├── dw_CY5_SLIDE001_01.tiff
│   ├── dw_DAPI_SLIDE001_01.tiff
│   ├── FOV_plots
│   │   ├── dw_CY5_SLIDE001_01.png
│   │   ├── dw_DAPI_SLIDE001_01.png
│   │   └── mask_dw_DAPI_SLIDE001_01.tiff.png
│   ├── log_files
│   │   ├── conversion_SLIDE001_01.log
│   │   ├── dw_CY5_SLIDE001_01.tiff.log.txt
│   │   ├── dw_DAPI_SLIDE001_01.tiff.log.txt
│   │   ├── dw_SLIDE001_01.log
│   │   ├── PSF_1.25_1.406_438.0_284.0_300.0.tiff.log.txt
│   │   ├── PSF_1.25_1.406_684.0_284.0_300.0.tiff.log.txt
│   │   └── PSF_SLIDE001_01.log
│   ├── masks
│   │   └── mask.01.tiff
│   └── PSF
│       ├── PSF_1.25_1.406_438.0_284.0_300.0.tiff
│       └── PSF_1.25_1.406_684.0_284.0_300.0.tiff
├── TEST_preprocessing_0.log
├── TEST_segmentation_GPU.log
└── test_SLIDE001_01.nd2
```

After running the profile estimation a folder called \*_profiles_output_SLIDE001 is created containing the .tsv files outputs of all the channels. To plot the profiles copy this folder in the plot_profiles folder and run the scripts as explained in the next paragraph. 
```
├── SLIDE001
│   ├── CY5_SLIDE001_01.tiff
│   ├── DAPI_SLIDE001_01.tiff
│   ├── dw_CY5_SLIDE001_01.tiff
│   ├── dw_DAPI_SLIDE001_01.tiff
│   ├── FOV_plots
│   │   ├── dw_CY5_SLIDE001_01.png
│   │   ├── dw_DAPI_SLIDE001_01.png
│   │   └── mask_dw_DAPI_SLIDE001_01.tiff.png
│   ├── log_files
│   │   ├── conversion_SLIDE001_01.log
│   │   ├── dw_CY5_SLIDE001_01.tiff.log.txt
│   │   ├── dw_DAPI_SLIDE001_01.tiff.log.txt
│   │   ├── dw_SLIDE001_01.log
│   │   ├── PSF_1.25_1.406_438.0_284.0_300.0.tiff.log.txt
│   │   ├── PSF_1.25_1.406_684.0_284.0_300.0.tiff.log.txt
│   │   └── PSF_SLIDE001_01.log
│   ├── masks
│   │   └── mask.01.tiff
│   ├── PSF
│   │   ├── PSF_1.25_1.406_438.0_284.0_300.0.tiff
│   │   └── PSF_1.25_1.406_684.0_284.0_300.0.tiff
│   └── TEST_profiles_output_SLIDE001
│       ├── mean_intensity_profiles_CY5.tsv
│       ├── mean_intensity_profiles_DAPI.tsv
│       ├── median_intensity_profiles_CY5.tsv
│       ├── median_intensity_profiles_DAPI.tsv
│       ├── nuclei_stats_CY5.tsv
│       └── nuclei_stats_DAPI.tsv
├── TEST_preprocessing_0.log
├── TEST_profile.log
├── TEST_segmentation_GPU.log
└── test_SLIDE001_01.nd2
```
To plot the profiles refer to the README.md in the plot_profiles folder.
