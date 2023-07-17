#!/bin/bash

while [[ $# -gt 0 ]]
do
  case "$1" in
    --path_raw_folder)
      path_raw_folder="$2"
      shift 
      shift 
      ;;
  esac
done

for output_dir in $(find "${path_raw_folder}"/* -maxdepth 0 -type d -exec echo {} \;)
do 
    mkdir -p "${output_dir}"/log_files
    mv "${output_dir}"/*.log* "${output_dir}"/log_files
    
    if ls "${output_dir}"/PSF_* 1> /dev/null 2>&1
    then 
      mkdir -p "${output_dir}"/PSF
      mv "${output_dir}"/PSF_* "${output_dir}"/PSF
    fi
done
