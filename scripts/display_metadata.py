from extract_metadata import *

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Extract deconvolution metadata from image')
    parser.add_argument('--path_to_raw', type=str,
                        help='Absolute path to the raw folder')
    args = parser.parse_args()

    absolute_paths_to_images = [f"{args.path_to_raw}/{i}" for i in os.listdir(args.path_to_raw)]

    for path in absolute_paths_to_images:
        basename = os.path.basename(path)
        extension = basename.split(".")[-1]

        if extension == "nd2": 
            obj = metadata_reader()
            obj.ND2(path)
            metadata = obj.extract_metadata_nd2(output = False)

        elif extension == "czi": 
            obj = metadata_reader()
            obj.CZI(path)
            obj.extract_metadata_czi(output = False)

        else: 
            raise Exception('Processing for files different from nd2 and czi is not implemented :S')