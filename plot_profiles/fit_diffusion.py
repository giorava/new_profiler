import torch
import pyro
from torch import nn
from torch.distributions import constraints
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
                        help = "SLIDEID of the slide to fit with timepoint define in time")
    parser.add_argument("--clip_infletion", type = str, 
                        help = "(True/False) Whether to clip the profiles at the infletion point (be carefull)")
    args = parser.parse_args()

    path_to_profiles = args.path_to_clipped_profiles
    time = float(args.time)
    slide_ID = args.SLIDEID
    clip = args.clip_infletion
    
    
    ### load the profiles
    with open(path_to_profiles, 'rb') as fp:
        clipped_profiles = pickle.load(fp)
        
    ### retrieve the data for the slide to fit
    key = [i for i in clipped_profiles.keys() if re.search(slide_ID, i)][0]
    selected_data = clipped_profiles[key]
    #r_clipped = selected_data["d_clipped"]
    #yfish_data = selected_data["points_prof_clipped"]/np.max(selected_data["points_prof_clipped"])
    r_before_clipping_inversion = selected_data["d_clipped"]
    yfish_before_clipping_inversion = selected_data["points_prof_clipped"]/np.max(selected_data["points_prof_clipped"])

    ## fitting polynomial
    Z = np.polyfit(r_before_clipping_inversion,  yfish_before_clipping_inversion, deg = 5)
    pol = np.poly1d(Z)
    r = np.linspace(0, 1, 1000)
    y_pred = pol(r)

    ## computing the gradients
    if clip=="True": 
        grad1 = np.gradient(y_pred)
        grad2 = np.gradient(grad1)
        infls = np.where(np.diff(np.sign(grad2)))[0]       
        if len(infls)==0: 
            logging.info(' No infletion points found')
            r_clipped = r_before_clipping_inversion
            yfish_data = yfish_before_clipping_inversion
        else: 
            logging.info(' Clipping based on infletion point')
            r_point = r[infls]
            r_infletion_data = np.where(r_before_clipping_inversion>=r_point)[0][0]
            r_clipped = r_before_clipping_inversion[r_infletion_data:]
            yfish_data_unnorm = yfish_before_clipping_inversion[r_infletion_data:]
            r_clipped = np.linspace(0, 1, len(yfish_data_unnorm))
            yfish_data = yfish_data_unnorm/np.max(yfish_data_unnorm)  
    else: 
        logging.info(' No clipping based on infletion point')
        r_clipped = r_before_clipping_inversion
        yfish_data = yfish_before_clipping_inversion
        
    
    ### split datasets and plot some stuff
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
    initial_guesses =  10**np.linspace(-60, 60, int(1000))
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

    #### bayesian regression with pyro
    x = torch.linspace(0.001, 1, len(yfish_data))
    y = torch.tensor(yfish_data, dtype=torch.float32)
    t = torch.tensor(time, dtype=torch.float32)
    
    def model(r, YFISH_data, timepoint):
        # Define priors       
        sigma = pyro.sample("sigma", dist.Normal(loc = 1, scale = 1))
        D_prior = pyro.sample("D_coeff", dist.Normal(loc = min_results[0], scale = min_results[0]*0.5))

        # Define likelihood
        mu = diffusion(r=r, D=D_prior, t=timepoint)
        pyro.sample("yfish_pred", dist.Normal(mu, sigma), obs=YFISH_data)

    # Perform inference step with Markov Chain Monte Carlo
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=10, num_chains=1)
    mcmc.run(r=x, YFISH_data=y, timepoint=t)

    # retrieve the D_app
    trace = mcmc.get_samples()
    D_app_dist = trace["D_coeff"]
    np.savetxt("posterior_D_coeff.txt", D_app_dist)
    
    # bootstrap to estimate the coefficient
    samples = []
    for i in range(5000):
        samp = np.random.choice(D_app_dist, len(D_app_dist)//4, replace=False)
        samples.append(samp)        
    mean_dist = [np.mean(i) for i in samples]   
    mean_of_mean = np.mean(mean_dist) 
    
    for D_app_samp in D_app_dist: 
        r = np.linspace(0.0001, 1, len(yfish_data))
        predict_ = diffusion(r = r, D = D_app_samp, t = time)
        plt.plot(r, predict_, color = "gray", alpha = 0.1)
        
    #ypredict = diffusion(r = r, D = D_app_dist, t = time)
    plt.scatter(r_clipped, yfish_data, color = "k")
    #plt.plot(r, ypredict, color = "red")
    
    plt.show()
    plt.savefig("Bayesian_regression")
    plt.close()
        
    