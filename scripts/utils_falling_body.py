# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:47:07 2021

@author: halvorak
"""
import numpy as np
import scipy.stats
#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import matplotlib.pyplot as plt



def ode_model_plant(t, x, w, par):
    
    #Unpack states and parameters
    rho_0 = par["rho_0"]
    M = par["M"]
    g = par["g"]
    a = par["a"]
    k = par["k"]
    

    #Allocate space and write the model
    x_dot = np.zeros(x.shape)
    x_dot[0] = x[1] + w[0]
    x_dot[1] = rho_0 * np.exp(-x[0] / k) * np.square(x[1]) * x[2]/ 2 - g + w[1]
    x_dot[2] = w[2] #it is equal to w[2]
    return x_dot

def hx(x, v, par):
    y = np.sqrt(np.square(par["M"]) + np.square(x[0] - par["a"])) + v
    return y

def fx_for_UT_gen_Q(sigmas, list_of_keys, t_span, x, par, w):
    
    for i in range(len(list_of_keys)):
        key = list_of_keys[i]
        if not key in par:
            raise KeyError("This key should be in par")
        par[key] = sigmas[i]
    yi = fx_ukf_ode(ode_model_plant,
                    t_span,
                    x,
                    args_ode = (w, par)
                    )
    # print(yi)
    return yi
        
    

def fx_ukf_ode(ode_model, t_span, x0, args_ode = None, args_solver = {}):
    res = scipy.integrate.solve_ivp(ode_model,
                                    t_span,
                                    x0,
                                    args = args_ode,
                                    **args_solver)
    x_all = res.y
    x_final = x_all[:, -1]
    return x_final

def get_x0_literature():
    x0 = np.array([300e3,  
                   -20e3, 
                   0.001 
                   ])
    return x0

def get_literature_values():
    #Nominal parameter values
    par_mean = {"rho_0": 2., # [lb-sec^2/ft4]
                "k": 20e3, # [ft]
                "g": 32.2, # [ft/s2]
                "M": 100e3, # [ft]
                "a": 100e3 # [ft]
                }
    
    #Kalman filter values in the description
    Q = np.diag([0., 0., 0.]) #as in Dan Simon's exercise text
    #Can try different Q-values
    Q = np.diag([1e4, 1e4, 1e-8]) 
    # Q = np.diag([1e-8, 1e-8, 1e-8]) 
    # Q = np.eye(4)*1e-3
    
    #Measurement noise
    R = np.array([10e3]) #[ft^2]
    
    return par_mean, Q, R


def get_param_ukf_case1(std_dev_prct = 0.05, plot_dist = True):
    """
    Generates gamma distributions for the parameters. Mean of gamma dist = par_mean from get_literature_values, and standard deviation of gamma dist = mean_literature*std_dev_prct

    Parameters
    ----------
    std_dev_prct : TYPE, optional float
        DESCRIPTION. The default is 0.05. Percentage of standard deviation compared to mean value of parameter
    plot_dist : TYPE, optional bool
        DESCRIPTION. The default is True. Whether or not to plot the distributions

    Returns
    -------
    par_dist : TYPE, dict
        DESCRIPTION. Each key cntains a gamma distribution, with mean = literature mean and std_dev = literature standard dev
    par_det : TYPE, dict
        DESCRIPTION. Key contains deterministic parameters.

    """
    
    par_mean, Q, R = get_literature_values()
    par_dist = {}
    for (key, val) in par_mean.items():
        if key == "g":
            continue
        alpha, loc, beta = get_param_gamma_dist(val, val*std_dev_prct, num_std = 2)
        par_dist[key] = scipy.stats.gamma(alpha, loc = loc, scale = 1/beta)
        
        
    par_det = {"g": par_mean["g"]}
    
    if plot_dist:
        dim_dist = len(par_dist)
        fig, ax = plt.subplots(dim_dist, 1)
        i = 0
        for key, dist in par_dist.items():
            #compute the mode numerically, as dist.mode() is not existing
            mode = scipy.optimize.minimize(lambda x: -dist.pdf(x),
                                           dist.mean(),
                                           tol = 1e-10)
            mode = mode.x
            print(key, mode, dist.mean())
            x = np.linspace(dist.ppf(.001), dist.ppf(.999), 100)
            ax[i].plot(x, dist.pdf(x), label = "pdf")
            ax[i].set_ylabel(key)
            ylim = ax[i].get_ylim()
            ax[i].plot([par_mean[key], par_mean[key]], [ylim[0], ylim[1]], 
                    label = "nom")
            ax[i].plot([dist.mean(), dist.mean()], [ylim[0], ylim[1]], 
                    linestyle = "dashed", label = "Mean")
            ax[i].plot([dist.mean()*(1-std_dev_prct), dist.mean()*(1-std_dev_prct)], [ylim[0], ylim[1]], 
                    label = "Mean-std_lit")
            ax[i].plot([dist.mean() - dist.std(), dist.mean() - dist.std()], [ylim[0], ylim[1]], 
                    linestyle = "dashed", label = "Mean-std_gamma")
            ax[i].plot([mode, mode], [ylim[0], ylim[1]], label = "Mode")
            # ax[i].plot([dist.mode(), dist.mode()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean")
            ax[i].set_ylim(ylim)
            ax[i].legend()
            i += 1
            
        return par_dist, par_det, fig, ax
    else:
        fig, ax = None, None
        return par_dist, par_det, fig, ax
    

def get_param_gamma_dist(mean, std_dev, num_std = 3):
    
    ##For making (mean_gamma = mean) AND (var_gamma = std_dev**2)
    loc = mean - std_dev*num_std
    alpha = num_std**2
    beta = num_std/std_dev
    
    #For putting (mode= mean-std_dev) AND 
    # loc = mean - std_dev*num_std
    # beta = 1/std_dev
    # # alpha = num_std#*std_dev**2
    # alpha = beta**2*std_dev**2#*std_dev**2
    return alpha, loc, beta

def get_sigmapoints_and_weights(par_dist):
    """
    Returns sigma points and weights for the distributions in the container par_dist.

    Parameters
    ----------
    par_dist : TYPE list, dict, a container which is iterable by for loop. len(par_dist) = n
        DESCRIPTION. each element contains a scipy.dist

    Returns
    -------
    sigmas : TYPE np.array((n, (2n+1)))
        DESCRIPTION. (2n+1) sigma points
    w : TYPE np.array(2n+1,)
        DESCRIPTION. Weight for every sigma point

    """
    
    n = len(par_dist) #dimension of parameters
    
    # Compute the required statistics
    mean = np.array([dist.mean() for k, dist in par_dist.items()])
    var = np.diag([dist.var() for k, dist in par_dist.items()])
    cm3 = np.array([scipy.stats.moment(dist.rvs(size = int(1e6)), moment = 3) 
                    for k, dist in par_dist.items()]) #3rd central moment
    cm4 = np.array([scipy.stats.moment(dist.rvs(size = int(1e6)), moment = 4) 
                    for k, dist in par_dist.items()]) #4th central moment
    
    #Generate sigma points
    sigma_points = spc.GenUTSigmaPoints(n) #initialize the class
    s, w = sigma_points.compute_scaling_and_weights(var,  #generate scaling and weights
                                                    cm3, 
                                                    cm4)
    sigmas, P_sqrt = sigma_points.compute_sigma_points(mean, #sigma points and P_sqrt
                                                        var, 
                                                        s)
    return sigmas, w


