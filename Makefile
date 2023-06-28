include *.config
export

path_bin="$(shell pwd)/bin"

interactive_session: 
	srun --nodes=1 --mem=30GB --tasks-per-node=1 --partition=cpu-interactive --pty /bin/bash 

install_python_dependencies: profiler.yml
	conda env create -f profiler.yml

copy_raw_data: 
	bash scripts/copy_raw_data.sh \
		--source_raw_folder_path ${source_raw_folder_path} \
		--path_raw_folder ${path_raw_folder}

show_metadata: 
	python scripts/display_metadata.py --path_to_raw ${path_raw_folder}

run_preprocessing: 
	echo ">>> submitting PREPROCESSING ${expID} <<<"       
	sbatch \
		--partition=cpuq \
		--array=0-$$(( ${number_of_images}-1 )) \
		--cpus-per-task=${threads} \
		--mem=${memory_per_image} \
		--job-name="pre${expID}" \
		--time=${preprocessing_estimated_time} \
		--output=${path_raw_folder}/${expID}_preprocessing_%a.log \
		--export=path_bin=$(path_bin),path_raw_folder=${path_raw_folder},dw_iterations=${dw_iterations},threads=${threads},path_raw_folder=${path_raw_folder},perform_decolvolution=${perform_decolvolution} \
		scripts/preprocessing.sh
    
after_preproc_cleaning: 
	bash scripts/after_preproc_cleaning.sh --path_raw_folder ${path_raw_folder} 

plot_fovs: 
	bash scripts/plot_fovs.sh --path_raw_folder ${path_raw_folder} --channel_name ${dapi_channel_name} 
	bash scripts/plot_fovs.sh --path_raw_folder ${path_raw_folder} --channel_name ${yfish_channel_name} 

run_segmentation_GPU: 
	echo ">>> submitting SEGMENTATION ${expID} <<<"       
	sbatch \
		--partition=gpuq  \
		--gres=gpu:1 \
		--cpus-per-gpu=${threads} \
		--mem=${memory_per_image} \
		--job-name="s${expID}" \
		--time="${segmentation_estimated_time}" \
		--output=${path_raw_folder}/${expID}_segmentation_GPU.log \
		--export=path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}",estimated_nuc_diameter="${estimated_nuc_diameter}",use_dw_dapi="${use_dw_dapi}",standardize_image_for_seg="${standardize_image_for_seg}" \
		scripts/segmentation.sh

run_segmentation_CPU: 
	echo ">>> submitting SEGMENTATION ${expID} <<<"       
	sbatch \
		--partition=cpuq  \
		--cpus-per-task=${threads} \
		--mem=${memory_per_image} \
		--job-name="s${expID}" \
		--time="${segmentation_estimated_time}" \
		--output=${path_raw_folder}/${expID}_segmentation_CPU.log \
		--export=path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}",estimated_nuc_diameter="${estimated_nuc_diameter}",use_dw_dapi="${use_dw_dapi},standardize_image_for_seg="${standardize_image_for_seg}"" \
		scripts/segmentation.sh

run_profile: 
	echo ">>> submitting PROFILEs COMPUTATION ${expID} <<<"       
	sbatch \
		--partition=cpuq \
		--mem=36GB \
		--job-name="p${expID}" \
		--time="${segmentation_estimated_time}" \
		--output=${path_raw_folder}/${expID}_profile.log \
		--export=expID=${expID},path_raw_folder="${path_raw_folder}",dapi_channel_name="${dapi_channel_name}",yfish_channel_name="${yfish_channel_name}",pixel_dx="${pixel_dx}",pixel_dy="${pixel_dy}",pixel_dz="${pixel_dz}" \
		scripts/profiles.sh

after_run_cleaning:
	bash scripts/after_run_cleaning.sh --path_raw_folder ${path_raw_folder} 
	mv *.log "${path_raw_folder}"

