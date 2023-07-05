#!/bin/bash 
while [[ $# -gt 0 ]]
do
  case "$1" in
    --source_raw_folder_path)
      source_raw_folder_path="$2"
      shift 
      shift
      ;;
    --path_raw_folder)
      path_raw_folder="$2"
      shift 
      shift
      ;;
  esac
done


if [[ "${source_raw_folder_path}" == "${path_raw_folder}" ]]
then
	echo "SOURCE and DEST are the same" 
else
  mkdir -p ${path_raw_folder}
  cp -v "${source_raw_folder_path}"/* "${path_raw_folder}"
fi