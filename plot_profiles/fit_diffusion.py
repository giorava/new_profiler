import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.mcmc.util import initialize_model
import numpy as np 
import matplotlib.pyplot as plt
import pickle, re
from scipy.optimize import minimize, Bounds, curve_fit, basinhopping
import argparse, tqdm
import logging
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO)

def diffusion(r, D, t):
    
    """
    Solution of the diffusion equation on spherical coordinates 
    and with the boundaries conditions defined in Eriks notes
    
    Parameters:
        r (np.ndarray): numpy array of radial positions
        D (float): estimated diffusion constant
        t (float): time point value

    Returns:
        diffusion_profile (torch.Tensor): array of diffusion values 
    
    """
    
    # conver arguments to torch
    r = torch.tensor(r, dtype=torch.float32)
    D = torch.tensor(D, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)

    sum_factor = torch.zeros_like(r)

    for k in range(1, 500):
        lam_k = (k * np.pi) ** 2
        phi_k = (1 / r) * torch.sin(k * np.pi * r)

        sum_factor += ((-1) ** (k + 1)) * torch.exp(-D * lam_k * t) * (phi_k) / (k * np.pi)

    diff = 1 - 2 * sum_factor

    ## flip around x
    flipped_d = torch.linspace(1, 0, len(diff))
    _, sorting = torch.sort(flipped_d)
    diffusion_profile = diff[sorting]
    
    return diffusion_profile

def cost_function(D, distance_array, yfish_array, time_array): 
    
    """
    Cost Function to fit the yfish data with the diffusion model
    
    Parameters: 
        D (float): current estimate for the diffusion constant
        yfish_array (list[np.ndarray]): list of yfish profiles (NOT normalized to max)
        time_array (list[float]): list of timepoints to fit

    Returns:
        cost (float): MSE of the current fit with respect to the YFISH data
    """
    
    cost = 0 
    for i, time in enumerate(time_array): 
        
        yfish = yfish_array[i]
        r_data = distance_array[i]
        r = np.linspace(0.001, 1, len(yfish))
        predicted = diffusion(r = r,
                              D = D,
                              t = time).numpy()
        
        interpolated_predicted = np.interp(r_data, r, predicted)
        
        n = len(yfish) # number of radius points
        mse = (1/n)*np.sum((yfish-interpolated_predicted)**2)
        cost += mse 
    
    return cost

def minimization_step(distance_array, yfish_array, time_array, initial_guess):
    
    """
    Minimization of the cost function to fit the YFISH data 
    
    Parameters: 
        yfish_array (list[np.ndarray]): list of yfish profiles (NOT normalized to max)
        time_array (list[float]): list of timepoints to fit
        initial_guess (float): an initial guess for the fit

    Returns:
        D_values (list[float]): list of predicted apparent diffusion coefficients
        convergence (list[bool]): list of boolean value indicating whether the algorithm converged or not for 
                                  the corresponding D value
    """

    global cost_function
    

    minimizer_kwargs = {
        "args": (distance_array, yfish_array, time_array), 
        "bounds": [(0,None)]
    }
    results = basinhopping(func = cost_function,
                            x0 = initial_guess,
                            minimizer_kwargs = minimizer_kwargs)
    
    return results["x"][0], results["success"]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Estimate the apparent diffusion constant for GPSeq score calculation \
    using Bayesian regression")
    parser.add_argument("--path_to_clipped_profiles", 
                        help = "Absolute path to the clipped data .pkl generated during the profile computation")
    parser.add_argument("--time", type = str, 
                        help = "Time point in seconds")
    parser.add_argument("--SLIDEID", type = str, 
                        help = "SLIDEID () of the slide to fit with timepoint define in time")
    args = parser.parse_args()

    path_to_profiles = args.path_to_clipped_profiles
    time = float(args.time)
    slide_ID = args.SLIDEID
    
    
    ### load the profiles
    with open(path_to_profiles, 'rb') as fp:
        clipped_profiles = pickle.load(fp)
        
    ### retrieve the data for the slide to fit
    key = [i for i in clipped_profiles.keys() if re.search(slide_ID, i)][0]
    selected_data = clipped_profiles[key]
    r_clipped = selected_data["d_clipped"]
    yfish_data = selected_data["points_prof_clipped"]/np.max(selected_data["points_prof_clipped"])
    mult_factor = np.max(selected_data["points_prof_clipped"])
    
    indexs = np.arange(0, len(r_clipped))
    index_train, index_test = train_test_split(indexs, test_size = 0.2)
    X_train, X_test = r_clipped[index_train], r_clipped[index_test]
    Y_train, Y_test = yfish_data[index_train], yfish_data[index_test]
    plt.scatter(X_train, Y_train, color = "green")
    plt.scatter(X_test, Y_test, color = "red")
    plt.show()
    plt.savefig("test")
    plt.close()
    
    ### Brute force optimization to find the initial guess (the mse is convex)
    logging.info(' BRUTE optimization to find the initial guess')
    initial_guesses =  10**np.linspace(-60, 60, int(100))
    cost = []
    for guess in tqdm.tqdm(initial_guesses): 
        _cost = cost_function(D = guess, distance_array = [X_train],
                              yfish_array = [Y_train], time_array = [time])
        cost.append(_cost)
    cost = np.array(cost)
    min_cost = np.where(cost==np.min(cost))[0][0]
    opt_initial_guess = initial_guesses[min_cost]
    
    ### fit with minimize with different initial values
    logging.info(' BASINHOPPING for local fit')
    min_results = minimization_step(distance_array = [X_train],
                                    yfish_array = [Y_train],
                                    time_array = [time],
                                    initial_guess = opt_initial_guess)
    
    ### evaluate fit on test set 
    print(opt_initial_guess, min_results)
    r = np.linspace(0.0001, 1, len(yfish_data))
    ypredict = diffusion(r = r, D = min_results[0] , t = time)
    plt.scatter(X_test, Y_test, color = "red")
    plt.scatter(X_train, Y_train, color = "green")
    plt.plot(r, ypredict, color = "red")
    
    test_error = cost_function(D = min_results[0], distance_array = [X_test], yfish_array = [Y_test], time_array = [time])
    training_error = cost_function(D = min_results[0], distance_array = [X_train], yfish_array = [Y_train], time_array = [time])
    
    interpolated_Y_test = np.interp(X_test, r, ypredict)
    interpolated_Y_train = np.interp(X_train, r, ypredict)
    plt.vlines(X_test, interpolated_Y_test, Y_test , 
               label = f"Test set error: {round(test_error, 4)}", color = "red")
    plt.vlines(X_train, interpolated_Y_train, Y_train, 
               label = f"Training set error: {round(training_error, 4)}", color = "green")
    
    plt.xlabel("Clipped distance from lamina")
    plt.ylabel("Normalized YFISH signal to maximum")  
    plt.title(f"Best fit {slide_ID}, timepoint = {time/60} min \n Apparent Diffusion constant = {min_results[0]}")
    plt.legend()
    plt.show()
    plt.savefig("test_minimize")
    plt.close()
    
    
    #### implement bayesian regression with pyro on those
    
    