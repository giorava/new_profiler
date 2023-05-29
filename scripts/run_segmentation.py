import segmentation_dapi
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
    args = parser.parse_args()

    logging.info(f" PROCESSING {args.image_folder}")
    seg_obj = segmentation_dapi.DapiSegmentation(
        image_folder = args.image_folder, 
        dapi_channel_name = args.dapi_channel_name, 
        dx = float(args.dx), dy = float(args.dy), dz = float(args.dz), 
        nuclei_dimension = int(args.estimated_nuc_diameter)
    )
    seg_obj.run_folder()

