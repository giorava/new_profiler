import matplotlib.pyplot as plt 
import numpy as np
import os, argparse, re
import tifffile
import logging
import warnings 
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="plotting the fovs to select the nice ones for the profile estimation")
    parser.add_argument('--image_folder', type = str,
                    help = "Absolute path to the folder with the preprocessed images")
    parser.add_argument('--YFISH_channel_name', type = str,
                    help = "Name of the YFISH channel")
    args = parser.parse_args()

    logging.info(f" Plotting FOVs in {args.image_folder}")
    deconvolved_yfish_images = [f"{args.image_folder}/{f}" for f in os.listdir(args.image_folder) if re.match(f"dw_{args.YFISH_channel_name}+.*\.tiff", f)]
    not_dw_yfish_images = [f"{args.image_folder}/{f}" for f in os.listdir(args.image_folder) if re.match(f"{args.YFISH_channel_name}+.*\.tiff", f)]
    if len(deconvolved_yfish_images)==0: 
        yfish_images = not_dw_yfish_images
    elif len(deconvolved_yfish_images)!=0: 
        yfish_images = deconvolved_yfish_images
    else: 
        warnings.warn("TIFF files not found, double check extensions and names")

    if not os.path.isdir(f"{args.image_folder}/FOV_plots"):
        os.mkdir(f"{args.image_folder}/FOV_plots")

    for file in yfish_images: 

        logging.info(f"Plotting {file}")

        base = os.path.basename(file)
        fov_idx = base.split(".")[0].split("_")[-1]
        slide_idx = re.search("(SLIDE)\d+(?:\d)?", base).group()

        img_array = tifffile.tifffile.TiffFile(file).asarray()
        mid_z = img_array.shape[0]//2
        
        fig, axs = plt.subplots(figsize = (20, 20))
        image_mid_z = img_array[mid_z, :, :]
        plt.imshow(image_mid_z, cmap = "bone", vmin = 0, vmax = np.max(image_mid_z)*1.3)
        
        plt.title(f"{base}, {fov_idx}", fontsize = 20)
        if len(deconvolved_yfish_images)==0:
            plt.savefig(f"{args.image_folder}/FOV_plots/{args.YFISH_channel_name}_{slide_idx}_{fov_idx}.png")
        elif len(deconvolved_yfish_images)>0: 
            plt.savefig(f"{args.image_folder}/FOV_plots/dw_{args.YFISH_channel_name}_{slide_idx}_{fov_idx}.png")
        plt.close()

    logging.info(f"--------")

