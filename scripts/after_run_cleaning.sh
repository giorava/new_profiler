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

mkdir -p "${path_raw_folder}"/processed 
mkdir -p "${path_raw_folder}"/raw	

for i in "${path_raw_folder}"/*SLIDE*
do 
  if [ -d ${i} ] ; then 
    mv -v ${i} "${path_raw_folder}"/processed
  fi 

  if [ -f ${i} ] ; then 
    mv -v ${i} "${path_raw_folder}"/raw
  fi
done

