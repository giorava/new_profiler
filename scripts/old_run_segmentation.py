import segmentation_dapi
import argparse
import logging

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description = "Run segmentation on dapi images")
    parser.add_argument('--image_folder', type = str,
                        help = "Absolute path to the folder with the preprocessed images")
    parser.add_argument('--dapi_channel_name', type = str,
                        help = "Name of the dapi channel")
    parser.add_argument('--sigma_gaussian', type = str,
                        help = "gaussian smoothing value, default (0.5)")
    parser.add_argument('--gamma_adjust', type = str, 
                        help = "gamma correction factor for luminosity, default 1")
    args = parser.parse_args()

    logging.info(f" PROCESSING {args.image_folder}")
    seg_obj = segmentation_dapi.DapiSegmentation(
        image_folder = args.image_folder, 
        dapi_channel_name = args.dapi_channel_name, 
        sigma_gaussian = float(args.sigma_gaussian), 
        gamma_adjust = float(args.gamma_adjust)
    )
    seg_obj.run_folder()

