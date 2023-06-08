#!/bin/bash 

while :
do
  case "$1" in
    --source_raw_folder_path )
      source_raw_folder_path="$2"
      shift 2
      ;;
    --path_raw_folder )
      path_raw_folder="$2"
      shift 2
      ;;
  esac
done

if [[ "${source_raw_folder_path}" == "${path_raw_folder}" ]]
then
	echo "SOURCE and DEST are the same" 
else
    cp -v "${source_raw_folder_path}"/* "${path_raw_folder}"
fi