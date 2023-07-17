#!/bin/bash

while [[ $# -gt 0 ]]
do
  case "$1" in
    --path_raw_folder)
      path_raw_folder="$2"
      shift 
      shift 
      ;;
    --channel_name)
      channel_name="$2"
      shift
      shift
      ;;  
  esac
done

eval "$(conda shell.bash hook)"
conda activate new_profiler

for folder in "${path_raw_folder}"/*SLIDE*
do 
  if [ -d ${folder} ] ; then 
    python scripts/plot_fovs.py --image_folder ${folder} --YFISH_channel_name ${channel_name}
  fi 
done