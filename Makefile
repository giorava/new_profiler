include *.config
export

install_python_dependencies: profiler.yml
	conda env create -f profiler.yml

check_external_dependencies: 
	echo "to be implemented for deconwolved and bfconvert"

copy_raw_data: 
	bash scripts/copy_raw_data.sh \
		--source_raw_folder_path ${source_raw_folder_path} \
		--path_raw_folder ${path_raw_folder}

run_preprocessing: 
	echo ">>> submitting PREPROCESSING ${expID} <<<"       
	sbatch \
		--partition=cpuq \
		--array=0-$((${number_of_images}-1)) \
		--cpus-per-task=${threads} \
		--mem=${memory_per_image} \
		--job-name="pre${expID}" \
		--time=${preprocessing_estimated_time} \
		--export=path_raw_folder=${path_raw_folder},dw_iterations=${dw_iterations},threads=${threads},path_raw_folder=${path_raw_folder} \
		scripts/preprocessing_cluster.sh
    
after_preproc_cleaning: 
	bash scripts/after_preproc_cleaning.sh --path_raw_folder ${path_raw_folder} 


run_segmentation: 
	echo ">>> submitting SEGMENTATION ${expID} <<<"       
	sbatch \
		--partition=cpuq  \
		--cpus-per-task=${threads} \
		--mem=${memory_per_image} \
		--job-name="s${expID}" \
		--time="${segmentation_estimated_time}" \
		--export=path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}",estimated_nuc_diameter="${estimated_nuc_diameter}",use_dw_dapi="${use_dw_dapi}" \
		scripts/segmentation.sh

run_profile: 
	echo ">>> submitting PROFILEs COMPUTATION ${expID} <<<"       
	sbatch \
		--partition=cpuq \
		--mem=36GB \
		--job-name="p${expID}" \
		--time="${segmentation_estimated_time}" \
		--export=path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",yfish_channel_name="${yfish_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}" \
		scripts/profiles.sh

after_run_cleaning:
	bash scripts/after_run_cleaningsh --path_raw_folder ${path_raw_folder} 
	mv *.log "${path_raw_folder}"
