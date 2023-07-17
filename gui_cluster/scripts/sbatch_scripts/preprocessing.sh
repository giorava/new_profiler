#!/bin/bash                     

# Load Modules and set env variables
source ${HOME}/.bashrc
eval "$(conda shell.bash hook)"
conda activate new_profiler
module load python/3.9.10
module load fftw/3.3.10-intel-oneapi-mpi-2021.4.0
PATH=$PATH:${path_bin}   
PATH=$PATH:${path_bin}/bftools

declare -a paths_raw_image=()
for path_raw_image in "${path_raw_folder}"/*
do 
    if [[ -f ${path_raw_image} ]]; then 
        if [[ ${path_raw_image} == *.nd2 ]]; then 
            paths_raw_image+=( "${path_raw_image}" )
        fi
        if [[ ${path_raw_image} == *.czi ]]; then 
            paths_raw_image+=( "${path_raw_image}" )
        fi
    fi
done

path_raw_image=${paths_raw_image[$SLURM_ARRAY_TASK_ID]}
echo ">>>> Processing Image ${path_raw_image}"

python scripts/run_process_custom.py \
    --path_raw_image "${path_raw_image}" \
    --dw_iterations "${dw_iterations}" \
    --threads "${threads}" \
    --perform_decolvolution "${perform_decolvolution}"
    
