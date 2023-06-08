import argparse
import logging
import process_custom_class
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__": 

    ### parse the path to raw images folder
    parser = argparse.ArgumentParser(description='Perform all the steps included deconvolution')
    parser.add_argument('--path_raw_image', type=str,
                        help='Absolute path to the raw image (nd2 or czi)')
    parser.add_argument('--dw_iterations', type=str,
                        help='Number of iterations')
    parser.add_argument('--threads', type=str,
                        help='number of threads')
    args = parser.parse_args()

    obj = process_custom_class.ProcessCustom(path_raw_image = args.path_raw_image, 
                                             dw_iterations = args.dw_iterations, 
                                             threads = args.threads)
    
    obj.run()

