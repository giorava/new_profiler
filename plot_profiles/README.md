# Plot profiles
This folder contains a set of scripts to filter the output data from the preprocessing pipeline and plot the fluorescence intensity profiles. 

### Install the dependencies

To install the dependencies follow these instructions: 
```
conda env create -f pyro_env.yml
```

### Run G1 selection
The first step to perform is copying the output folders of RadiantKit2.0 to this folder. Then to perform the G1 selection run: 
```
python filter_G1.py --expID "TEST" --dapi_ch_name "dapi" --yfish_ch_name "cy5"
```
This script performs debris removal using K-means cluster and performs G1 selection on the higher cluster using a Gaussian Mixture model with 2 components on integrated intensity and area. The most abundant cell are kept.

### Run profiles plotting


### Run diffusion fitting
