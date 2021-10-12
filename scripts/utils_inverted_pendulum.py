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

def ode_model_plant(x, t, u, par, sin_fx = None, cos_fx = None):
    if sin_fx is None:
        sin_fx = np.sin
    
    if cos_fx is None:
        cos_fx = np.cos
    
    #Unpack states and parameters
    d, d_dot, theta, theta_dot = x
    m = par["m"]
    M = par["M"]
    g = par["g"]
    B = par["B"]
    l = par["l"]
    r = par["r"]
    J = np.multiply(m, np.square(r))/2
    
    #Allocate space and write the model
    x_dot = np.zeros(x.shape)
    x_dot[0] = d_dot # dd/dt
    x_dot[2] = theta_dot # d theta/dt
    # x_dot[1] = 
    x_dot[3] = (m*g*l*sin_fx(theta)*(M + m) - 
                m*l*cos_fx(theta)*(u + m*l*np.square(theta_dot)*sin_fx(theta) - B*d_dot)) #numerator for d theta_dot/dt
    x_dot[3] = x_dot[3] / ((J + m*np.square(l))*(M + m) -
                           np.square(m*l)*np.square(cos_fx(theta)))
    
    x_dot[1] = (u - m*l*x_dot[3]*cos_fx(theta) + m*l*np.square(theta_dot)*sin_fx(theta) - B*d_dot) / (M + m)
    
    return x_dot

def get_x0_inverted_pendulum():
    x0 = np.array([0., # d [rad] 
                   0., # d_dot [rad/s]
                   .1, # theta [rad]
                   0.  # theta_dot [rad/s]
                   ])
    return x0

def get_nominal_values():
    #Nominal parameter values
    par_nom = {"m": .2, # [kg]
               "M": 1., # [kg]
               "g": 9.81, # [m/s2]
               "B": .1, # [N/(m/s)]
               "l": 1., # [m]
               "r": .02, # [m]
               }
    
    #Kalman filter values in the description
    Q = np.diag([0., 4e-4, 0., 4e-2])
    R = np.square(.1) #[m^2]
    
    return par_nom, Q, R


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
    mean_ut, var_ut = ut.unscented_transform(sigmas_g, w, fx = fx)
    
    #Monte Carlo simulation
    N_mc = [int(1e2), int(5e2), int(1e3), int(2e3), int(5e3), int(1e4), int(1e5), int(5e5), int(1e6)]#, int(5e6)]#, int(1e7)]
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