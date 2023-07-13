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
export PATH=${PATH}:/${path_to_gui_folder}     ## can also add this to .bashrc
```
If you are not using the HT cluster please make sure that the binaries in /bin are working: 
```
${path_to_bin}/dw --help
${path_to_bin}/dw_bw --help
${path_to_bin}/bftools/bfconvert --help
```
you should see the help page of each dependency.


## Running with the GUI

To run the analysis using the GUI you have to follow the following preliminary steps where HT_USER is your user name: 

1. log into the cluster using X11 forwarding:
    ```
    ssh -YC ${HT_USER}@hpclogin.fht.org
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
    cd ${path_to_gui_folder}
    python GUI.py
    ```

The following window will appear: 

<br>
<figure>  
<p align="center">
<img src="https://github.com/giorava/new_profiler/assets/107054086/db8e873b-91ea-4ae3-8371-aceb5692d245" width="900"></a>
<figcaption>  
  <p align="center">
    GUI window.
  </p>
</figcaption>
</p>
</figure>

After choosing the path of your raw images with `browse` you can complete the other fields with the help of `Show Metadata`. Remember to save the configuration. To submit the preprocessing just press `Submit preprocessing` and follow the submission in the command line. To check the status of the job press `Show queue`. After the preprocessing is completed clean the folders and plot the FOVs. Before proceeding with segmentation double check the nuclei diameter by looking at the FOV plots of dapi and change if necessary. Submit the segmentation with `Submit segmentation`. Analogously, after segmentation computes the profiles with `Submit profiler` and cleans the target folders with `After run cleaning` to have a nicely organized output. The output .tsv will be present in `${path_to_raw}/${processed}/${SLIDEID}/${expID}_profiles_output_${SLIDEID}`.

> **Warning**
> With FOVs with very few cells the segmentation might fail with the default options. You might see some over splitting. In this case, try:
> 
> 1. Increasing the nuclei diameter
> 2. Setting `Standardize image for segmentation` to True
> 3. Setting use deconvolved DAPI to False
>    
> Please try only one of these options at the time. If the problem persists consider using an alternative segmentation procedure (to be implemented soon).

## Output structure

before preprocessing
```
└── test_SLIDE001_01.nd2
```
after preprocessing, cleaning and FOVs plotting
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

after segmentation 
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

after profile estimation
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


## Running outside the HT computational cluster


## Running as a command line tool
