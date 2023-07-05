from profiler_classes.segmentation_dapi import *
import argparse
import logging

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description = "Run segmentation on dapi images")
    parser.add_argument('--image_folder', type = str,
                        help = "Absolute path to the folder with the preprocessed images")
    parser.add_argument('--dapi_channel_name', type = str,
                        help = "Name of the dapi channel")
    parser.add_argument('--dx', type = str, 
                        help = "lateral dimension of the pizel (nm)")
    parser.add_argument('--dy', type = str, 
                        help = "lateral dimension of the pizel (nm)")
    parser.add_argument('--dz', type = str, 
                        help = "axial dimension of the pizel (nm)")
    parser.add_argument('--estimated_nuc_diameter', type = str, 
                        help = "estimate diameter of nuclei in pixels")
    parser.add_argument('--use_dw_dapi', type = str, 
                        help = "whether or not to use dw dapi for deconvolution (default False)")
    parser.add_argument('--standardize_image_for_seg', type = str, 
                        help = "whether or not to standardize the image for segmentation (default False)")
    args = parser.parse_args()

    logging.info(f" PROCESSING {args.image_folder}")
    seg_obj = DapiSegmentation(
        image_folder = args.image_folder, 
        dapi_channel_name = args.dapi_channel_name, 
        dx = float(args.dx), dy = float(args.dy), dz = float(args.dz), 
        nuclei_dimension = int(args.estimated_nuc_diameter),
        use_dw_dapi = args.use_dw_dapi,
        standardize_image_for_seg = args.standardize_image_for_seg
    )
    seg_obj.run_folder()

