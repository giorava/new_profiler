import argparse
import logging
from scripts.profiler_classes.process_custom_class import *
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
    parser.add_argument("--perform_decolvolution", type = str, 
                        help="True if you want to perform the deconvolution, False otherwise")
    args = parser.parse_args()

    obj = ProcessCustom(path_raw_image = args.path_raw_image, 
                                             dw_iterations = args.dw_iterations, 
                                             threads = args.threads, 
                                             perform_decolvolution = args.perform_decolvolution)
    
    obj.run()

