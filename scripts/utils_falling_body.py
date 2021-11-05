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
import matplotlib

font = {'size': 14}

matplotlib.rc('font', **font)


def ode_model_plant(t, x, w, par):
    
    #Unpack states and parameters
    rho_0 = par["rho_0"]
    # M = par["M"]
    g = par["g"]
    # a = par["a"]
    k = par["k"]
    

    #Allocate space and write the model
    x_dot = np.zeros(x.shape)
    x_dot[0] = x[1] + w[0]
    x_dot[1] = rho_0 * np.exp(-x[0] / k) * np.square(x[1]) * x[2]/ 2 - g + w[1]
    x_dot[2] = w[2] #it is equal to w[2]
    return x_dot

def hx(x, par):
    y_rep = par["y_rep"] #this is the same as the measurment noise, v
    y = np.sqrt(np.square(par["M"]) + np.square(x[0] - par["a"])) + y_rep
    return np.array([y])

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

def fx_for_UT_gen_wbar_Q(sigmas, list_of_keys, t_span, x, par, w):
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

def hx_for_UT_gen_R(sigmas, list_of_keys, x, par):
    
    for i in range(len(list_of_keys)):
        key = list_of_keys[i]
        if not key in par:
            raise KeyError("This key should be in par")
        par[key] = sigmas[i]
    yi = hx(x, par)
    # yi = np.array([hx(x, par)])
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
    par_mean_fx = {"rho_0": 2., # [lb-sec^2/ft4]
                "k": 20e3, # [ft]
                "g": 32.2 # [ft/s2]
                }
    par_mean_hx = {
        "M": 100e3, # [ft]
        "a": 100e3 # [ft]
        }
    
    #Kalman filter values in the description
    Q = np.diag([0., 0., 0.]) #as in Dan Simon's exercise text
    #Can try different Q-values
    Q = np.diag([1e3, 1e3, 1e-8]) 
    # Q = np.diag([1e3, 1e3, 1e-8]) #used 
    # Q = np.diag([1e-8, 1e-8, 1e-8]) 
    # Q = np.eye(4)*1e-3
    
    #Measurement noise
    R = np.array([10e3]) #[ft^2]
    
    return par_mean_fx, par_mean_hx, Q, R

def compute_performance_index_valappil(x_kf, x_ol, x_true):
    J = np.divide(np.linalg.norm(x_kf - x_true, axis = 1, ord = 2),
                  np.linalg.norm(x_ol - x_true, axis = 1, ord = 2))
    return J

def get_param_ukf_case1(par_fx = True, std_dev_prct = 0.05, plot_dist = True):
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
    par_dist = {}
    par_det = {}
    
    if par_fx: #want parameters for state equations
        par_mean, par_hx, Q, R = get_literature_values()
    else: #want parameters for measurement equation
        par_func, par_mean, Q, R = get_literature_values()
    
    
    for (key, val) in par_mean.items():
        if key == "g":
            continue
        alpha, loc, beta = get_param_gamma_dist(val, val*std_dev_prct, num_std = 2)
        # print(f"{key}: {alpha}, {loc}, {beta}")
        par_dist[key] = scipy.stats.gamma(alpha, loc = loc, scale = 1/beta)
        
    if "g" in par_mean.keys():   
        par_det["g"] = par_mean["g"]
    
    if plot_dist:
        # par_dist.pop("k")
        dim_dist = len(par_dist)
        fig, ax = plt.subplots(dim_dist, 1)
        if dim_dist == 1:
            ax = [ax]
        i = 0
        
        for key, dist in par_dist.items():
            #compute the mode numerically, as dist.mode() is not existing
            mode = scipy.optimize.minimize(lambda x: -dist.pdf(x),
                                           dist.mean(),
                                           tol = 1e-10)
            mode = mode.x
            # print(key, mode, dist.mean())
            x = np.linspace(dist.ppf(1e-5), dist.ppf(.999), 100)
            ax[i].plot(x, dist.pdf(x), label = "pdf")
            # ax[i].set_xlabel("Air density at sea level, " + r"$\rho_0$" + r" ($lb-s^2/ft^4$)")
            ax[i].set_xlabel(key)
            ax[i].set_ylabel("pdf")
            ylim = ax[i].get_ylim()
            
            # ax[i].plot([par_mean[key], par_mean[key]], [ylim[0], ylim[1]], 
            #         label = "nom")
            # ax[i].plot([dist.mean(), dist.mean()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean")
            # ax[i].plot([dist.mean()*(1-std_dev_prct), dist.mean()*(1-std_dev_prct)], [ylim[0], ylim[1]], 
            #         label = "Mean-std_lit")
            # ax[i].plot([dist.mean() - dist.std(), dist.mean() - dist.std()], [ylim[0], ylim[1]], 
            #         linestyle = "dashed", label = "Mean-std_gamma")
            # ax[i].plot([mode, mode], [ylim[0], ylim[1]], label = "Mode")
            
            
            ax[i].scatter(par_mean[key], dist.pdf(par_mean[key]), label = r"$\mu_\Gamma = \mu_{lit} = \theta_{UKF}$")
            ax[i].scatter([par_mean[key] - dist.std(), par_mean[key] + dist.std()], 
                              [dist.pdf(par_mean[key] - dist.std()), dist.pdf(par_mean[key] + dist.std())], label = r"$\mu_\Gamma \pm \sigma_\Gamma = \mu_{lit} \pm \sigma_{lit}$")
            ax[i].scatter(mode, dist.pdf(mode), label = r"$\theta_{true}$")
            ndist = 2
            ax[i].scatter(dist.mean() - ndist*dist.std(), 
                          dist.pdf(dist.mean() - ndist*dist.std()), 
                          label = r"$\mu_{lit} - 2\sigma_{lit}$")
            
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
    
    # print(f"mean: {mean}\n",
    #       f"var: {var}\n",
    #       f"cm3: {cm3}\n",
    #       f"cm4: {cm4}")
    
    #Generate sigma points
    sigma_points = spc.GenUTSigmaPoints(n) #initialize the class
    s, w = sigma_points.compute_scaling_and_weights(var,  #generate scaling and weights
                                                    cm3, 
                                                    cm4)
    sigmas, P_sqrt = sigma_points.compute_sigma_points(mean, #sigma points and P_sqrt
                                                        var, 
                                                        s)
    return sigmas, w

def get_lhs_points(dist_dict, N_lhs_dist = 10, plot_mc_samples = False, labels = None):
    """
    

    Parameters
    ----------
    dist_dict : TYPE dict
        DESCRIPTION. {"a": scipy.stats.gamma(alpha, beta),
                      "b": scipy.stats.lognormal(mu, cov)}
    N_lhs_dist : TYPE, optional iny
        DESCRIPTION. The default is 10. number of LHS samples
    plot_mc_samples : TYPE, optional bool
        DESCRIPTION. The default is False. Plot or not
    labels : TYPE, optional list
        DESCRIPTION. The default is None.

    Returns
    -------
    x_lhs : TYPE
        DESCRIPTION.
    sample : TYPE
        DESCRIPTION.
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    dim_dist = len(dist_dict)
    sampler = scipy.stats.qmc.LatinHypercube(d = dim_dist)
    # mean_lhs = np.zeros(N_lhs_dist))
    # var_lhs = np.zeros(len(N_lhs_dist))
    # for k in range(len(N_lhs)):
    sample = sampler.random(N_lhs_dist) #samples are done in the CDF (cumulative distribution function)
        
    #convert to values in the state domain
    x_lhs = {}
    x_rand = {}
    i = 0
    for key, dist in dist_dict.items():
        x_lhs[key] = dist.ppf(sample[:, i])#.reshape(-1,1)
        x_rand[key] = dist.rvs(size = N_lhs_dist)
        i += 1
    if not plot_mc_samples:
        fig, ax = (None, None)
    else: #do plot
        if labels is None:
            labels = ["" for k in range(dim_dist)]
        
        #A plot of how the points distribute in the pdf of the dist
        fig_pdf, ax_pdf = plt.subplots(dim_dist, 1)
        if dim_dist == 1:
            ax_pdf = [ax_pdf]
        # labels = [r"$\rho$ $[kg/m^3]$", "d [m]"]
        line_color = []
        i = 0
        for key, dist in dist_dict.items(): #plot the pdf of each distribution
            x_dummy = np.linspace(dist.ppf(.001), dist.ppf(.999), 100)    
            l = ax_pdf[i].plot(x_dummy, dist.pdf(x_dummy), label = "pdf")
            ax_pdf[i].scatter(x_lhs[key], np.zeros(N_lhs_dist), 
                          label = "LHS",
                          marker = "x")
            # ax_pdf[i].scatter(x_rand[key], np.zeros(N_lhs_dist), label = "rvs")
            ax_pdf[i].set_xlabel(labels[i])
            ax_pdf[i].set_ylabel("pdf")
            i += 1
            
        #Also a plot of the CDF + LHS
        fig_cdf, ax_cdf = plt.subplots(dim_dist, 1)
        if dim_dist == 1:
            ax_cdf = [ax_cdf]
        grid_cdf = np.linspace(0,.999,num = N_lhs_dist+1)
        i = 0
        for key, dist in dist_dict.items(): #plot the pdf of each distribution
            x_dummy = np.linspace(dist.ppf(.001), dist.ppf(.999), 100)
            x_cdf = dist.ppf(grid_cdf)
            l = ax_cdf[i].plot(x_dummy, dist.cdf(x_dummy), label = "CDF")
            scat_lhs = ax_cdf[i].scatter(x_lhs[key], sample[:, i], marker= "x", label = "LHS sample")
            # ax_cdf[i].scatter(x_rand[key], dist.cdf(x_rand[key]), label = "rvs")
            xlim = ax_cdf[i].get_xlim()
            ylim = ax_cdf[i].get_ylim()
            
            for j in range(len(grid_cdf)):
                grid_x = np.array([xlim[0], x_cdf[j], x_cdf[j]])
                grid_y = np.array([grid_cdf[j], grid_cdf[j], ylim[0]])
                line_lhs_grid, = ax_cdf[i].plot(grid_x, grid_y, linestyle = "dashed", color = 'r')#, label = "LHS grid")
            line_lhs_grid.set_label("LHS grid")
            ax_cdf[i].set_ylabel("CDF")
            ax_cdf[i].set_xlabel(labels[i])
            i += 1
        
    
        fig = [fig_pdf, fig_cdf]
        ax = [ax_pdf, ax_cdf]
        for axi in ax:
            try:
                axi[0].legend()
            except:
                axi.legend()
    return x_lhs, sample, fig, ax
def get_mc_points(dist_dict, N_mc_dist = 1000, plot_mc_samples = False, labels = None):
    """
    

    Parameters
    ----------
    dist_dict : TYPE dict
        DESCRIPTION. {"a": scipy.stats.gamma(alpha, beta),
                      "b": scipy.stats.lognormal(mu, cov)}
    N_mc_dist : TYPE, optional int
        DESCRIPTION. The default is 1000. number of MC samples
    plot_mc_samples : TYPE, optional bool
        DESCRIPTION. The default is False. Plot or not
    labels : TYPE, optional list
        DESCRIPTION. The default is None.

    Returns
    -------
    x_lhs : TYPE
        DESCRIPTION.
    sample : TYPE
        DESCRIPTION.
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    """
    dim_dist = len(dist_dict)
    x_mc = {}
    i = 0
    for key, dist in dist_dict.items():
        x_mc[key] = dist.rvs(size = N_mc_dist)
        # x_rand[key] = dist.rvs(size = N_mc_dist)
        i += 1
    if not plot_mc_samples:
        fig_pdf, ax_pdf = (None, None)
    else: #do plot
        if labels is None:
            labels = ["" for k in range(dim_dist)]
        
        #A plot of how the points distribute in the pdf of the dist
        fig_pdf, ax_pdf = plt.subplots(dim_dist, 1)
        if dim_dist == 1:
            ax_pdf = [ax_pdf]
        # labels = [r"$\rho$ $[kg/m^3]$", "d [m]"]
        line_color = []
        i = 0
        for key, dist in dist_dict.items(): #plot the pdf of each distribution
            x_dummy = np.linspace(dist.ppf(.001), dist.ppf(.999), 100)    
            l = ax_pdf[i].plot(x_dummy, dist.pdf(x_dummy), label = "pdf")
            ax_pdf[i].scatter(x_mc[key], np.zeros(N_mc_dist), 
                          label = "MC",
                          marker = "x")
            # ax_pdf[i].scatter(x_rand[key], np.zeros(N_lhs_dist), label = "rvs")
            ax_pdf[i].set_xlabel(labels[i])
            ax_pdf[i].set_ylabel("pdf")
            i += 1
            
        for axi in ax_pdf:
            try:
                axi[0].legend()
            except:
                axi.legend()
    return x_mc, fig_pdf, ax_pdf

# def get_wmean_Q_from_lhs(par_lhs, x0, t_span, w_plant, par_nom):
#     x_nom = fx_ukf_ode(ode_model_plant, t_span, x0, args_ode = (w_plant, par_nom))
#     N_mc = list(par_lhs.values())[0].shape[0] #the number of MC samples (or LHS)
#     dim_x = x0.shape[0]
#     x_stoch = np.zeros((dim_x, N_mc))
#     par_i = {}
#     for i in range(N_mc):
#         for key, val in par_lhs.items():
#             par_i[key] = val[i]
#         x_stoch[:, i] = fx_ukf_ode(ode_model_plant, t_span, x0, args_ode = (w_plant, par_i))
    
#     Q_lhs = np.cov(x_stoch)
#     w_mean = np.mean(x_stoch - x_nom, axis = 1)
#     return w_mean, Q_lhs

def get_w_realizations_from_mc(par_mc, x0, t_span, w_plant, par_nom):
    x_nom = fx_ukf_ode(ode_model_plant, t_span, x0, args_ode = (w_plant, par_nom))
    N_mc = list(par_mc.values())[0].shape[0] #the number of MC samples (or LHS)
    dim_x = x0.shape[0]
    x_stoch = np.zeros((dim_x, N_mc))
    par_i = {}
    for i in range(N_mc):
        for key, val in par_mc.items():
            par_i[key] = val[i]
        x_stoch[:, i] = fx_ukf_ode(ode_model_plant, t_span, x0, args_ode = (w_plant, par_i))
    w_stoch = x_stoch - x_nom.reshape(-1,1)
    return w_stoch

def get_wmean_Q_from_mc(par_mc, x0, t_span, w_plant, par_nom):
    
    w_stoch = get_w_realizations_from_mc(par_mc, x0, t_span, w_plant, par_nom)
    w_mean = np.mean(w_stoch, axis = 1)
    Q = np.cov(w_stoch)
    return w_mean, Q

def get_wmode_Q_from_mc(par_mc, x0, t_span, w_plant, par_nom, nbins = 20):
    
    dim_x = x0.shape[0]
    w_mode = np.zeros(dim_x)
    w_stoch = get_w_realizations_from_mc(par_mc, x0, t_span, w_plant, par_nom)
    #find the mode by maximizing the histogram values
    for i in range(dim_x):
        hist, bin_edges = np.histogram(w_stoch[i, :], bins = nbins)
        idx = np.argmax(hist) #returns the index where there are most samples in the bin
        mode_limits = np.array([bin_edges[idx], bin_edges[idx+1]]) #the mode is somewhere within these limits
        w_mode[i] = np.mean(mode_limits)
    Q = np.cov(w_stoch)
    return w_mode, Q


# def get_vmean_R_from_lhs(par_lhs, x, dim_y):
#     N_mc = list(par_lhs.values())[0].shape[0] #the number of MC samples (or LHS)
#     y = np.zeros((dim_y, N_mc))
#     par_i = {}
#     for i in range(N_mc):
#         for key, val in par_lhs.items():
#             par_i[key] = val[i]
#         y[:, i] = hx(x, par_i)
#     v_lhs = np.mean(y, axis = 1)
#     R_lhs = np.cov(y)
#     return v_lhs, R_lhs

def get_v_realizations_from_mc(par_mc, x, dim_y, par_nom):
    y_nom = hx(x, par_nom)
    N_mc = list(par_mc.values())[0].shape[0] #the number of MC samples (or LHS)
    y_stoch = np.zeros((dim_y, N_mc))
    par_i = {}
    for i in range(N_mc):
        for key, val in par_mc.items():
            par_i[key] = val[i]
        y_stoch[:, i] = hx(x, par_i)
    v_stoch = y_stoch - y_nom.reshape(-1,1)
    return v_stoch

def get_vmean_R_from_mc(par_mc, x, dim_y, par_nom):
    
    v_stoch = get_v_realizations_from_mc(par_mc, x, dim_y, par_nom)
    v_mean = np.mean(v_stoch, axis = 1)
    R = np.cov(v_stoch)
    return v_mean, R

def get_vmode_R_from_mc(par_mc, x, dim_y, par_nom, nbins = 20):
    
    v_stoch = get_v_realizations_from_mc(par_mc, x, dim_y, par_nom)
    hist, bin_edges = np.histogram(v_stoch, bins = nbins)
    idx = np.argmax(hist) #returns the index where there are most samples in the bin
    mode_limits = np.array([bin_edges[idx], bin_edges[idx+1]]) #the mode is somewhere within these limits
    v_mode = np.mean(mode_limits)
    R = np.cov(v_stoch)
    return v_mode, R




# def get_ys_by_MC(par_mc, x, dim_y):
#     N_mc = list(par_mc.values())[0].shape[0] #the number of MC samples (or LHS)
#     y = np.zeros((dim_y, N_mc))
#     par_i = {}
#     for i in range(N_mc):
#         for key, val in par_mc.items():
#             par_i[key] = val[i]
#         y[:, i] = hx(x, par_i)
    
#     return y

# def get_vmean_R_by_mc(par_mc, x, dim_y):
#     y_mat = get_ys_by_MC(par_mc, x, dim_y)
#     v_mean = np.mean(y_mat, axis = 1)
#     R = np.cov(y_mat)
#     # print(f"y_mat: {y_mat.shape}\n",
#     #       f"v_mean: {v_mean.shape}\n",
#     #       f"R: {R}\n")
#     return v_mean, R

def venturi_eq(q, rho, d, D, C, eps):
    beta = d/D
    y1 = np.multiply(rho, (1-np.power(beta,4)))/2
    y2 = np.multiply(np.square(d)*np.pi/4, C)
    y2 = np.multiply(y2, eps)
    y2 = np.square(np.divide(q, y2))
    y = np.multiply(y1, y2)/1e3 #[mbar]
    return np.array([y])
    
def uncertainty_venturi():
    
    #define the gamma distributions
    g = np.array([2, 2]) #"a" param in scipy.stats.gamma() dist
    loc = np.array([600, 0.0646*.95])
    scale = np.array([2, 1e-3])
    gamma_dist = [scipy.stats.gamma(g[i], loc = loc[i], scale = scale[i]) 
                  for i in range(len(g))]
    dist_dict = {"rho": gamma_dist[0]}
    get_lhs_points(dist_dict, plot_mc_samples=True, labels = [r"$\rho$ [kg/$m^3$]", "d [m]"])
    #Compute the central moments which is requried for GenUT
    mean_g = np.array([dist.mean() for dist in gamma_dist])
    var_g = np.diag([dist.var() for dist in gamma_dist])
    rvs = np.hstack([dist.rvs(size = int(1e6)).reshape(-1,1) 
                      for dist in gamma_dist]) #samples from the dist
    skew_g = scipy.stats.moment(rvs, moment = 3)
    kurt_g = scipy.stats.moment(rvs, moment = 4)
    
    #Compute sigma points in GenUT
    sigmas_gamma = spc.GenUTSigmaPoints(len(gamma_dist))
    s, w = sigmas_gamma.compute_scaling_and_weights(var_g, 
                                                    skew_g, 
                                                    kurt_g)
    sigmas_g, P_sqrt_g = sigmas_gamma.compute_sigma_points(mean_g, 
                                                            var_g, 
                                                            s)
    #Check sigma points are correct. The calculated mean and covariance should 
    #match moments of the original distributions
    q = 120/3600 #[m3/s]
    D = 146.36/1e3 #[m]
    eps = 1.
    C = 1.01 #[-]
    fx = lambda x: venturi_eq(q, x[0], x[1], D, C, eps)
    mean_ut, var_ut = ut.unscented_transformation(sigmas_g, w, fx = fx)
    
    #Monte Carlo simulation
    N_mc = [int(1e2), int(5e2), int(1e3), int(2e3), int(5e3), int(1e4), int(1e5), int(5e5), int(1e6), int(5e6)]#, int(1e7)]
    mean_mc = np.zeros(len(N_mc))
    var_mc = np.zeros(len(N_mc))
    for i in range(len(N_mc)):
        x_mc = np.hstack([dist.rvs(size = N_mc[i]).reshape(-1,1) 
                          for dist in gamma_dist])
        y_mc = fx(x_mc.T)
        mean_mc[i] = np.mean(y_mc)
        var_mc[i] = np.var(y_mc)
    
    #%% Latin hypercube sampling
    sampler = scipy.stats.qmc.LatinHypercube(d=len(gamma_dist))
    N_lhs = np.arange(10,510,20)
    mean_lhs = np.zeros(len(N_lhs))
    var_lhs = np.zeros(len(N_lhs))
    for k in range(len(N_lhs)):
        sample = sampler.random(N_lhs[k]) #samples are done in the CDF (cumulative distribution function)
        
        #convert to values in the state domain
        x_lhs = np.hstack([gamma_dist[i].ppf(sample[:, i]).reshape(-1,1) #ppf = percent point function
                           for i in range(len(gamma_dist))])
        y_lhs = fx(x_lhs.T)
        mean_lhs[k] = np.mean(y_lhs)
        var_lhs[k] = np.var(y_lhs)
    
    
    
    
    fig2, ax2 = plt.subplots(2,1)
    fig3, ax3 = plt.subplots(2,1)
    
    
    ax2[0].plot(N_mc, mean_mc, label = "MC", marker = "x")
    ax2[0].plot(N_mc, mean_ut*np.ones(len(N_mc)), label = "UT")
    
    ax2[1].plot(N_lhs, mean_lhs, label = "LHS")
    ax2[1].plot(N_lhs, mean_ut*np.ones(len(N_lhs)), label = "UT")
    
    
    ax3[0].plot(N_mc, var_mc, label = "MC", marker = "x")
    ax3[0].plot(N_mc, var_ut[0,0]*np.ones(len(N_mc)), label = "UT")
    ax3[1].plot(N_lhs, var_lhs, label = "LHS")
    ax3[1].plot(N_lhs, var_ut[0,0]*np.ones(len(N_lhs)), label = "UT")
    
    ax2[0].set_xscale("log")
    ax3[0].set_xscale("log")
    # ax2[0].set_xlabel("Number of MC evaluations")
    ax2[0].set_xlabel("Number of model evaluations")
    ax2[1].set_xlabel("Number of model evaluations")
    ax2[0].set_ylabel(r"$\mu_y$")
    ax2[1].set_ylabel(r"$\mu_y$")
    
    ax3[0].set_xlabel("Number of model evaluations")
    ax3[1].set_xlabel("Number of model evaluations")
    ax3[0].set_ylabel("Var(y)")
    ax3[1].set_ylabel("Var(y)")
    
    ax2[0].legend()
    ax2[1].legend()
    ax3[0].legend()
    ax3[1].legend()
    
    fig4, ax4 = plt.subplots(2,1)
    fig5, ax5 = plt.subplots(2,1)
    
    ax4[0].plot(N_mc, mean_mc/mean_ut*100, label = "MC", marker = "x")
    ax4[0].plot(N_mc, 100*np.ones(len(N_mc)), label = "UT")
    ax5[0].plot(N_mc, var_mc/var_ut[0,0]*100, label = "MC", marker = "x")
    ax5[0].plot(N_mc, 100*np.ones(len(N_mc)), label = "UT")
    
    ax4[1].plot(N_lhs, mean_lhs/mean_ut*100, label = "LHS", marker = "x")
    ax4[1].plot(N_lhs, 100*np.ones(len(N_lhs)), label = "UT")
    ax5[1].plot(N_lhs, var_lhs/var_ut[0,0]*100, label = "MC", marker = "x")
    ax5[1].plot(N_lhs, 100*np.ones(len(N_lhs)), label = "LHS")
    
    ax4[0].set_xscale("log")
    ax5[0].set_xscale("log")
    ax4[0].set_xlabel("Number of model evaluations")
    ax4[1].set_xlabel("Number of model evaluations")
    ax5[0].set_xlabel("Number of model evaluations")
    ax5[1].set_xlabel("Number of model evaluations")
    
    ax4[0].set_ylabel(r"$\mu/\mu_y$ [%]")
    ax4[1].set_ylabel(r"$\mu/\mu_y$ [%]")
    ax5[0].set_ylabel("Var(y) / Var(y, UT) [%]")
    ax5[1].set_ylabel("Var(y) / Var(y, UT) [%]")
    
    ax4[0].legend()
    ax4[1].legend()
    ax5[0].legend()
    ax5[1].legend()
    
    fig7, ax7 = plt.subplots(1,1)
    ax7.hist(y_mc.flatten(), bins = 50)
    ax7.set_xlabel(f"y-distribution, MC ({int(N_mc[-1])} simulations)")
    print(f"mean_mc: {mean_mc}\n",
          f"mean_UT: {mean_ut}\n",
          f"var_mc: {var_mc}\n",
          f"var_UT: {var_ut}")
    #%% pdf
    #Make plots of the marginal distributions (they are uncorrelated)
    fig, ax = plt.subplots(2,1)
    x = [np.linspace(dist.ppf(.001), dist.ppf(.999), 100) 
          for dist in gamma_dist]
    labels = [r"$\rho$ $[kg/m^3]$", "d [m]"]
    line_color = []
    for i in range(len(gamma_dist)): #plot the pdf of each distribution
        l = ax[i].plot(x[i], gamma_dist[i].pdf(x[i]), label = labels[i])
        ax[i].set_xlabel(labels[i])
        ax[i].set_ylabel("pdf")
        # line_color.append(l[0])
    conf_int = .95 #95% confidence interval
    conf_intervals = [dist.interval(conf_int) for dist in gamma_dist]
    # y_true = 
    
    #%% CDF
    fig6, ax6 = plt.subplots(2,1)
    y_cdf = [dist.cdf(xi) for (dist, xi) in zip(gamma_dist, x)]
    sample_lhs = sampler.random(N_lhs[0]) #samples are done in the CDF (cumulative distribution function)
        
    #convert to values in the state domain
    x_lhs = np.hstack([gamma_dist[i].ppf(sample_lhs[:, i]).reshape(-1,1) #ppf = percent point function
                       for i in range(len(gamma_dist))])
    
    grid_cdf = np.linspace(0,.999,num = N_lhs[0]+1)
    x_cdf = np.hstack([dist.ppf(grid_cdf).reshape(-1,1) #ppf = percent point function
                       for dist in gamma_dist])
    for i in range(len(gamma_dist)): #plot the pdf of each distribution
        l = ax6[i].plot(x[i], y_cdf[i])
        scat_lhs = ax6[i].scatter(x_lhs[:,i], sample_lhs[:, i], marker= "x", label = "LHS sample")
        xlim = ax6[i].get_xlim()
        ylim = ax6[i].get_ylim()
        for j in range(len(grid_cdf)):
            grid_x = np.array([xlim[0], x_cdf[j, i], x_cdf[j, i]])
            grid_y = np.array([grid_cdf[j], grid_cdf[j], ylim[0]])
            line_lhs_grid, = ax6[i].plot(grid_x, grid_y, linestyle = "dashed", color = 'r')#, label = "LHS grid")
        line_lhs_grid.set_label("LHS grid")
        ax6[i].set_xlabel(labels[i])
        ax6[i].set_ylabel("CDF")
        ax6[i].legend()#[scat_lhs, line_lhs_grid[0]])

    return None

def uncertainty_venturi2():
    
    #define the gamma distributions
    g = np.array([2, 2]) #"a" param in scipy.stats.gamma() dist
    loc = np.array([600, 0.0646*.95])
    scale = np.array([2, 1e-3])
    gamma_dist = [scipy.stats.gamma(g[i], loc = loc[i], scale = scale[i]) 
                  for i in range(len(g))]
    dist_dict = {"rho": gamma_dist[0]}
    get_lhs_points(dist_dict, plot_mc_samples=True, labels = [r"$\rho$ [kg/$m^3$]", "d [m]"])
    #Compute the central moments which is requried for GenUT
    mean_g = np.array([dist.mean() for dist in gamma_dist])
    var_g = np.diag([dist.var() for dist in gamma_dist])
    rvs = np.hstack([dist.rvs(size = int(1e6)).reshape(-1,1) 
                      for dist in gamma_dist]) #samples from the dist
    skew_g = scipy.stats.moment(rvs, moment = 3)
    kurt_g = scipy.stats.moment(rvs, moment = 4)
    
    #Compute sigma points in GenUT
    sigmas_gamma = spc.GenUTSigmaPoints(len(gamma_dist))
    s, w = sigmas_gamma.compute_scaling_and_weights(var_g, 
                                                    skew_g, 
                                                    kurt_g)
    sigmas_g, P_sqrt_g = sigmas_gamma.compute_sigma_points(mean_g, 
                                                            var_g, 
                                                            s)
    #Check sigma points are correct. The calculated mean and covariance should 
    #match moments of the original distributions
    q = 120/3600 #[m3/s]
    D = 146.36/1e3 #[m]
    eps = 1.
    C = 1.01 #[-]
    fx = lambda x: venturi_eq(q, x[0], x[1], D, C, eps)
    mean_ut, var_ut = ut.unscented_transformation(sigmas_g, w, fx = fx)
    
    #Monte Carlo simulation
    N_mc = [int(1e2), int(5e2), int(1e3), int(2e3), int(5e3), int(1e4), int(1e5), int(5e5), int(1e6), int(5e6)]#, int(1e7)]
    mean_mc = np.zeros(len(N_mc))
    var_mc = np.zeros(len(N_mc))
    for i in range(len(N_mc)):
        x_mc = np.hstack([dist.rvs(size = N_mc[i]).reshape(-1,1) 
                          for dist in gamma_dist])
        y_mc = fx(x_mc.T)
        mean_mc[i] = np.mean(y_mc)
        var_mc[i] = np.var(y_mc)
    
    #%% Latin hypercube sampling
    sampler = scipy.stats.qmc.LatinHypercube(d=len(gamma_dist))
    N_lhs = np.arange(10,510,20)
    mean_lhs = np.zeros(len(N_lhs))
    var_lhs = np.zeros(len(N_lhs))
    for k in range(len(N_lhs)):
        sample = sampler.random(N_lhs[k]) #samples are done in the CDF (cumulative distribution function)
        
        #convert to values in the state domain
        x_lhs = np.hstack([gamma_dist[i].ppf(sample[:, i]).reshape(-1,1) #ppf = percent point function
                           for i in range(len(gamma_dist))])
        y_lhs = fx(x_lhs.T)
        mean_lhs[k] = np.mean(y_lhs)
        var_lhs[k] = np.var(y_lhs)
    
    
    
    
    fig2, ax2 = plt.subplots(2,1)
    fig3, ax3 = plt.subplots(2,1)
    
    
    ax2[0].plot(N_mc, mean_mc, label = "MC", marker = "x")
    ax2[0].plot(N_mc, mean_ut*np.ones(len(N_mc)), label = "UT")
    
    ax3[0].plot(N_lhs, mean_lhs, label = "LHS")
    ax3[0].plot(N_lhs, mean_ut*np.ones(len(N_lhs)), label = "UT")
    
    
    ax2[1].plot(N_mc, var_mc, label = "MC", marker = "x")
    ax2[1].plot(N_mc, var_ut[0,0]*np.ones(len(N_mc)), label = "UT")
    ax3[1].plot(N_lhs, var_lhs, label = "LHS")
    ax3[1].plot(N_lhs, var_ut[0,0]*np.ones(len(N_lhs)), label = "UT")
    
    ax2[0].set_xscale("log")
    ax2[1].set_xscale("log")
    # ax2[0].set_xlabel("Number of MC evaluations")
    ax2[0].set_xlabel("Number of model evaluations")
    ax2[1].set_xlabel("Number of model evaluations")
    ax2[0].set_ylabel(r"$\mu_y$")
    ax3[0].set_ylabel(r"$\mu_y$")
    
    ax2[1].set_xlabel("Number of model evaluations")
    ax3[1].set_xlabel("Number of model evaluations")
    ax2[1].set_ylabel("Var(y)")
    ax3[1].set_ylabel("Var(y)")
    
    ax2[0].legend()
    ax2[1].legend()
    ax3[0].legend()
    ax3[1].legend()
    
    fig4, ax4 = plt.subplots(2,1)
    fig5, ax5 = plt.subplots(2,1)
    
    ax4[0].plot(N_mc, mean_mc/mean_ut*100, label = "MC", marker = "x")
    ax4[0].plot(N_mc, 100*np.ones(len(N_mc)), label = "UT")
    ax5[0].plot(N_mc, var_mc/var_ut[0,0]*100, label = "MC", marker = "x")
    ax5[0].plot(N_mc, 100*np.ones(len(N_mc)), label = "UT")
    
    ax4[1].plot(N_lhs, mean_lhs/mean_ut*100, label = "LHS", marker = "x")
    ax4[1].plot(N_lhs, 100*np.ones(len(N_lhs)), label = "UT")
    ax5[1].plot(N_lhs, var_lhs/var_ut[0,0]*100, label = "MC", marker = "x")
    ax5[1].plot(N_lhs, 100*np.ones(len(N_lhs)), label = "LHS")
    
    ax4[0].set_xscale("log")
    ax5[0].set_xscale("log")
    ax4[0].set_xlabel("Number of model evaluations")
    ax4[1].set_xlabel("Number of model evaluations")
    ax5[0].set_xlabel("Number of model evaluations")
    ax5[1].set_xlabel("Number of model evaluations")
    
    ax4[0].set_ylabel(r"$\mu/\mu_y$ [%]")
    ax4[1].set_ylabel(r"$\mu/\mu_y$ [%]")
    ax5[0].set_ylabel("Var(y) / Var(y, UT) [%]")
    ax5[1].set_ylabel("Var(y) / Var(y, UT) [%]")
    
    ax4[0].legend()
    ax4[1].legend()
    ax5[0].legend()
    ax5[1].legend()
    
    fig7, ax7 = plt.subplots(1,1)
    ax7.hist(y_mc.flatten(), bins = 50)
    ax7.set_xlabel(f"y-distribution, MC ({int(N_mc[-1])} simulations)")
    print(f"mean_mc: {mean_mc}\n",
          f"mean_UT: {mean_ut}\n",
          f"var_mc: {var_mc}\n",
          f"var_UT: {var_ut}")
    #%% pdf
    #Make plots of the marginal distributions (they are uncorrelated)
    fig, ax = plt.subplots(2,1)
    x = [np.linspace(dist.ppf(.001), dist.ppf(.999), 100) 
          for dist in gamma_dist]
    labels = [r"$\rho$ $[kg/m^3]$", "d [m]"]
    line_color = []
    for i in range(len(gamma_dist)): #plot the pdf of each distribution
        l = ax[i].plot(x[i], gamma_dist[i].pdf(x[i]), label = labels[i])
        ax[i].set_xlabel(labels[i])
        ax[i].set_ylabel("pdf")
        # line_color.append(l[0])
    conf_int = .95 #95% confidence interval
    conf_intervals = [dist.interval(conf_int) for dist in gamma_dist]
    # y_true = 
    
    #%% CDF
    fig6, ax6 = plt.subplots(2,1)
    y_cdf = [dist.cdf(xi) for (dist, xi) in zip(gamma_dist, x)]
    sample_lhs = sampler.random(N_lhs[0]) #samples are done in the CDF (cumulative distribution function)
        
    #convert to values in the state domain
    x_lhs = np.hstack([gamma_dist[i].ppf(sample_lhs[:, i]).reshape(-1,1) #ppf = percent point function
                       for i in range(len(gamma_dist))])
    
    grid_cdf = np.linspace(0,.999,num = N_lhs[0]+1)
    x_cdf = np.hstack([dist.ppf(grid_cdf).reshape(-1,1) #ppf = percent point function
                       for dist in gamma_dist])
    for i in range(len(gamma_dist)): #plot the pdf of each distribution
        l = ax6[i].plot(x[i], y_cdf[i])
        scat_lhs = ax6[i].scatter(x_lhs[:,i], sample_lhs[:, i], marker= "x", label = "LHS sample")
        xlim = ax6[i].get_xlim()
        ylim = ax6[i].get_ylim()
        for j in range(len(grid_cdf)):
            grid_x = np.array([xlim[0], x_cdf[j, i], x_cdf[j, i]])
            grid_y = np.array([grid_cdf[j], grid_cdf[j], ylim[0]])
            line_lhs_grid, = ax6[i].plot(grid_x, grid_y, linestyle = "dashed", color = 'r')#, label = "LHS grid")
        line_lhs_grid.set_label("LHS grid")
        ax6[i].set_xlabel(labels[i])
        ax6[i].set_ylabel("CDF")
        ax6[i].legend()#[scat_lhs, line_lhs_grid[0]])

    return None
