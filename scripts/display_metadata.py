from extract_metadata import *

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Extract deconvolution metadata from image')
    parser.add_argument('--path_to_image', type=str,
                        help='Absolute path to the image')
    args = parser.parse_args()

    basename = os.path.basename(args.path_to_image)
    extension = basename.split(".")[-1]

    if extension == "nd2": 
        obj = metadata_reader()
        obj.ND2(args.path_to_image)
        metadata = obj.extract_metadata_nd2(output = False)

    elif extension == "czi": 
        obj = metadata_reader()
        obj.CZI(args.path_to_image)
        obj.extract_metadata_czi(output = False)

    else: 
        raise Exception('Processing for files different from nd2 and czi is not implemented :S')
