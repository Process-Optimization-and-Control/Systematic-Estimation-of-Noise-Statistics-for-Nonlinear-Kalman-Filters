# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:14:40 2021

Falling body estimation. Example 14.2 in Dan Simon's book "Optimal State Estimation"

@author: halvorak
"""


import numpy as np
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
import colorcet as cc
import pathlib
import os
import scipy.linalg
import matplotlib.patches as plt_patches
import matplotlib.transforms as plt_transforms
import copy

# Did some modification to these packages
from myFilter import UKF
from myFilter import sigma_points as ukf_sp
# from myFilter import UKF_constrained

#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import utils_falling_body as utils_fb

#%% For running the sim N times
N = 1
dim_x = 3
j_valappil = np.zeros((dim_x, N))
j_valappil_lhs = np.zeros((dim_x, N))
j_valappil_qf = np.zeros((dim_x, N))
Ni = 0
while Ni < N:
    try:
        #%% Import the distributions of the parameters for the fx equations (states)
        #Modes of the dists are used for the true system, and mean of the dists are the parameters for the UKF
        
        # utils_ip.uncertainty_venturi()
        _, _, Q_nom, R_nom = utils_fb.get_literature_values()
        par_dist_fx, par_det_fx, fig_p_fx, ax_p_fx = utils_fb.get_param_ukf_case1(
            par_fx = True,
            std_dev_prct = 0.05, 
            plot_dist = True)
        
        mode_dist = np.zeros(len(par_dist_fx)) #for the mode of the distributions
        par_true_fx = par_det_fx.copy()
        par_kf_fx = par_det_fx.copy()
        del par_det_fx
        for key, dist in par_dist_fx.items():
            #Need to find the mode numericlly
            mode = scipy.optimize.minimize(lambda theta: -dist.pdf(theta),
                                           dist.mean(), #use mean as theta0
                                           tol = 1e-10) 
            par_true_fx[key] = mode.x #true system uses the mode values
            # par_true_fx[key] = dist.mean()+dist.std() #true system uses the mean - std_dev
            par_kf_fx[key] = dist.mean() #the ukf uses mean values reported in the literaure
        
            print(f"{key}_true/{key}_KF: {par_true_fx[key]/par_kf_fx[key]}")
        # par_kf_fx = par_true_fx
        #%% Import the distributions of the parameters for the fx equations (measurements)
        #Modes of the dists are used for the true system, and mean of the dists are the parameters for the UKF
        
        par_dist_hx, par_det_hx, fig_p_hx, ax_p_hx = utils_fb.get_param_ukf_case1(
            par_fx = False, #return based on par_true_hx in stead
            std_dev_prct = 0.05, 
            plot_dist = True)
        mode_dist = np.zeros(len(par_dist_hx)) #for the mode of the distributions
        par_true_hx = par_det_hx.copy()
        par_kf_hx = par_det_hx.copy()
        del par_det_hx
        for key, dist in par_dist_hx.items():
            #Need to find the mode numericlly
            mode = scipy.optimize.minimize(lambda theta: -dist.pdf(theta),
                                           dist.mean(), #use mean as theta0
                                           tol = 1e-10) 
            par_true_hx[key] = mode.x #true system uses the mode values
            # par_true_hx[key] = dist.mean() - dist.std() #true system uses the mean - std_dev
            par_kf_hx[key] = dist.mean() #the ukf uses mean values reported in the literaure
        
            print(f"{key}_true/{key}_KF: {par_true_hx[key]/par_kf_hx[key]}")
        # par_kf_hx = par_true_hx
        #Measurement equation also need y_repeatability
        # y_rep = scipy.stats.norm(loc = np.zeros(R_nom.shape[0]),
        #                          scale = np.sqrt(np.diag(R_nom))) #this is just 1D, so ok
        y_rep = scipy.stats.norm(loc = 0.,
                                 scale = np.sqrt(R_nom[0])) #this is just 1D, so ok
        par_true_hx["y_rep"] = y_rep.rvs()
        par_kf_hx["y_rep"] = y_rep.mean()
        par_dist_hx["y_rep"] = y_rep
        
        #%% Define samples for LHS
        N_lhs_dist = 10
        labels_fx = par_true_fx.copy()
        labels_fx.pop("g")
        par_lhs_fx, samples_lhs_fx, fig_lhs, ax_lhs = utils_fb.get_lhs_points(par_dist_fx,
                                                                      N_lhs_dist = N_lhs_dist,
                                                                      plot_mc_samples=True,
                                                                      labels = list(labels_fx.keys()))
        par_lhs_fx["g"] = np.ones(N_lhs_dist)*par_kf_fx["g"] #append with constant g
        par_lhs_hx, samples_lhs_hx, fig_lhs, ax_lhs = utils_fb.get_lhs_points(par_dist_hx, 
                                                                              N_lhs_dist = N_lhs_dist, 
                                                                              plot_mc_samples=True,
                                                                              labels = list(par_true_hx.keys())
                                                                              )
        
        #%% Define dimensions and initialize arrays
        # sigmas, w = utils_fb.get_sigmapoints_and_weights(par_dist)
        
        x0 = utils_fb.get_x0_literature()
        P0 = np.diag([3e8,#[ft^2], altitute, initial covariance matrix for UKF
                    4e6, # [ft^2], horizontal range
                    1e-6 # [?] ballistic coefficient
                    ])
        # print(x0+np.sqrt(np.diag(P0)))
        
        x0_kf = copy.deepcopy(x0) + np.sqrt(np.diag(P0)) #+ the standard deviation
        dim_x = x0.shape[0]
        dt_y = .5 # [s] <=> 5 ms. Measurement frequency
        dt_int = 1e-3 #[s] - discretization time for integration
        
        t_end = 30
        t = np.linspace(0, t_end, int(t_end/dt_y))
        dim_t = t.shape[0]
        # t = np.linspace(t_y[0],t_y[0+1], int(dt_y/dt), endpoint = True)
        
        y0 = utils_fb.hx(x0, par_true_hx)
        dim_y = y0.shape[0]
        y = np.zeros((dim_y, dim_t))
        y[:, 0] = y0
        
        x_true = np.zeros((dim_x, dim_t)) #[[] for _ in range(dim_t-1)] #make a list of list
        x_ol = np.zeros((dim_x, dim_t)) #Open loop simulation - same starting point and param as UKF
        x_post = np.zeros((dim_x, dim_t))
        x_post_lhs = np.zeros((dim_x, dim_t))
        x_post_qf = np.zeros((dim_x, dim_t))
        # x_prior = np.zeros((dim_x, dim_ty))
        
        x_true[:, 0] = x0
        x_post_lhs[:, 0] = x0_kf
        x_post[:, 0] = x0_kf
        x_post_qf[:, 0] = x0_kf
        x0_ol = copy.deepcopy(x0_kf)
        x_ol[:, 0] = x0_ol
        # x_prior[:, 0] = x0_kf
        t_span = (t[0],t[1])
        
        #%% Create noise
        # v = np.random.normal(loc = 0, scale = np.sqrt(R_nom), size = dim_t) #white noise
        # w_plant = np.random.multivariate_normal(np.zeros((dim_x,)), 
        #                                         Q_nom, 
        #                                         size = dim_t)
        w_plant = np.zeros((dim_t, dim_x))
        w_noise_kf = np.zeros(dim_x)
        # v_noise_kf = np.zeros(dim_y)
        
        #%% Define UKF with adaptive Q, R from UT
        points = ukf_sp.JulierSigmaPoints(dim_x,
                                          kappa = 3-dim_x)
        fx_ukf = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                     t_span, 
                                                     x,
                                                     args_ode = (w_noise_kf,
                                                                 par_kf_fx))
        
        hx_ukf = lambda x_in: utils_fb.hx(x_in, par_kf_hx)#.reshape(-1,1)
        
        #kf is where Q adapts based on UT of parametric uncertainty
        kf = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                       dim_z = dim_y, 
                                        dt = 100, 
                                        hx = hx_ukf, 
                                        fx = fx_ukf,
                                        points = points)
        kf.x = x_post[:, 0]
        kf.P = copy.deepcopy(P0)
        kf.Q = Q_nom #to be updated in a loop
        kf.R = R_nom #to be updated in a loop
        
        #%% Define UKF with adaptive Q, R from LHS/MC
        points_lhs = ukf_sp.JulierSigmaPoints(dim_x,
                                          kappa = 3-dim_x)
        fx_ukf_lhs = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                     t_span, 
                                                     x,
                                                     args_ode = (w_noise_kf,
                                                                 par_kf_fx))
        
        hx_ukf_lhs = lambda x_in: utils_fb.hx(x_in, par_kf_hx)#.reshape(-1,1)
        
        #kf is where Q adapts based on UT of parametric uncertainty
        kf_lhs = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                           dim_z = dim_y, 
                                            dt = 100, 
                                            hx = hx_ukf, 
                                            fx = fx_ukf,
                                            points = points_lhs)
        kf_lhs.x = x_post_lhs[:, 0]
        kf_lhs.P = copy.deepcopy(P0)
        kf_lhs.Q = Q_nom #to be updated in a loop
        kf_lhs.R = R_nom #to be updated in a loop
        
        #%% Define UKF with fixed Q
        points_qf = ukf_sp.JulierSigmaPoints(dim_x,
                                          kappa = 3-dim_x)
        fx_ukf_qf = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                     t_span, 
                                                     x,
                                                     args_ode = (w_noise_kf,
                                                                 par_kf_fx))
        
        hx_ukf_qf = lambda x_in: utils_fb.hx(x_in, par_kf_hx)#.reshape(-1,1)
        kf_qf = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                          dim_z = dim_y,
                                          dt = 100, 
                                          hx = hx_ukf_qf, 
                                          fx = fx_ukf_qf, 
                                          points = points_qf)
        kf_qf.x = x_post[:, 0]
        kf_qf.P = copy.deepcopy(P0)
        kf_qf.Q = Q_nom
        kf_qf.R = R_nom
        
        
        #%% Get parametric uncertainty of fx by GenUT
        sigmas_fx, w_fx = utils_fb.get_sigmapoints_and_weights(par_dist_fx)
        list_dist_fx_keys = list(par_dist_fx.keys()) # list of parameters with distribution
        fx_gen_Q = lambda si: utils_fb.fx_for_UT_gen_Q(si, 
                                                       list_dist_fx_keys, 
                                                       t_span, 
                                                       x0, 
                                                       par_kf_fx,
                                                       w_noise_kf)
        
        #%% Get parametric uncertainty of hx by GenUT
        sigmas_hx, w_hx = utils_fb.get_sigmapoints_and_weights(par_dist_hx)
        list_dist_hx_keys = list(par_dist_hx.keys()) # list of parameters with distribution
        hx_gen_R = lambda si: utils_fb.hx_for_UT_gen_R(si, 
                                                        list_dist_hx_keys, 
                                                        x0, 
                                                        par_kf_hx)
        
        
        #%% Simulate the plant and UKF
        for i in range(1,dim_t):
            t_span = (t[i-1], t[i])
            w_plant_i = w_plant[i, :]
            res = scipy.integrate.solve_ivp(utils_fb.ode_model_plant, 
                                            t_span,#(t_y[i-1],t_y[i]), 
                                            x_true[:, i-1], 
                                            # rtol = 1e-10,
                                            # atol = 1e-13
                                            args = (w_plant_i, par_true_fx)
                                            )
            x_true[:, i] = res.y[:, -1] #add the interval to the full list
            
            # Solve the open loop model prediction, based on the same info as UKF has (no measurement)
            res_ol = scipy.integrate.solve_ivp(utils_fb.ode_model_plant, 
                                               t_span,#(t_y[i-1],t_y[i]), 
                                               x_ol[:, i-1], 
                                               # rtol = 1e-10,
                                               # atol = 1e-13
                                               args = (w_noise_kf, par_kf_fx)
                                               )
            x_ol[:, i] = res_ol.y[:, -1] #add the interval to the full list
            
            #Prediction step of kalman filter. Make the functions for each UKF
            fx_ukf = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                          t_span, 
                                                          x,
                                                          args_ode = (w_noise_kf,
                                                                      par_kf_fx)
                                                          )
            fx_ukf_lhs = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                          t_span, 
                                                          x,
                                                          args_ode = (w_noise_kf,
                                                                      par_kf_fx)
                                                          )
            fx_ukf_qf = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                          t_span, 
                                                          x,
                                                          args_ode = (w_noise_kf,
                                                                      par_kf_fx)
                                                          )
            fx_gen_Q = lambda si: utils_fb.fx_for_UT_gen_Q(si, 
                                                           list_dist_fx_keys, 
                                                           t_span, 
                                                           x_post[:, i-1], 
                                                           par_kf_fx,
                                                           w_noise_kf)
            #Adaptive Q by UT
            mean_ut, Q_ut = ut.unscented_transformation(sigmas_fx, w_fx, fx = fx_gen_Q)
            kf.Q = Q_ut
            
            #Adaptive Q by LHS
            Q_lhs = utils_fb.get_Q_from_lhs(par_lhs_fx, 
                                            x_post_lhs[:, i-1], 
                                            t_span, 
                                            w_noise_kf)
            kf_lhs.Q = Q_lhs
            
            #Prediction step of each UKF
            kf.predict(fx = fx_ukf)
            kf_lhs.predict(fx = fx_ukf_lhs)
            kf_qf.predict(fx = fx_ukf_qf)
            # print(f"pred: {kf.x}")
            
            #Make a new measurement
            par_true_hx["y_rep"] = y_rep.rvs() #draw a new sample from noise statistics
            y[:, i] = utils_fb.hx(x_true[:, i], par_true_hx) #add the measurement
        
            #Adaptive R by UT
            x_prior = copy.deepcopy(kf.x_prior)
            hx_gen_R = lambda si: utils_fb.hx_for_UT_gen_R(si, 
                                                           list_dist_hx_keys, 
                                                           x_prior, 
                                                           par_kf_hx)
            mean_ut, R_ut = ut.unscented_transformation(sigmas_hx, 
                                                        w_hx, 
                                                        fx = hx_gen_R)
            kf.R = R_ut
            
            #Adaptive R by LHS
            x_prior_lhs = copy.deepcopy(kf_lhs.x_prior)
            R_lhs = utils_fb.get_R_from_lhs(par_lhs_hx, x_prior_lhs, dim_y)
            kf_lhs.R = R_lhs
            
            #Correction step of UKF
            kf.update(y[:, i], hx = hx_ukf)
            kf_lhs.update(y[:, i], hx = hx_ukf_lhs)
            kf_qf.update(y[:, i], hx = hx_ukf_qf)
            
            # Save the estimates
            x_post[:, i] = kf.x
            x_post_lhs[:, i] = kf_lhs.x
            x_post_qf[:, i] = kf_qf.x
            
        
        y[:, 0] = np.nan #the 1st measurement is not real, just for programming convenience
        
        
        
        #%% Compute performnance index
        j_valappil[:, Ni] = utils_fb.compute_performance_index_valappil(x_post, 
                                                                     x_ol, 
                                                                     x_true)
        j_valappil_lhs[:, Ni] = utils_fb.compute_performance_index_valappil(x_post_lhs, 
                                                                         x_ol, 
                                                                         x_true)
        j_valappil_qf[:, Ni] = utils_fb.compute_performance_index_valappil(x_post_qf, 
                                                                        x_ol, 
                                                                        x_true)
        Ni += 1
    except BaseException as e:
        print(e)
        continue
#%% Plot
plot_it = True
if plot_it:
    ylabels = ["x1 [ft]", "x2 [ft/s]", "x3 []", "y []"]#
        
    fig1, ax1 = plt.subplots(dim_x + 1, 1, sharex = True)
    for i in range(dim_x): #plot true states and ukf's estimates
        ax1[i].plot(t, x_true[i, :], label = "True")
        ax1[i].plot(t, x_post[i, :], label = "UKF_UT")
        ax1[i].plot(t, x_post_lhs[i, :], label = f"UKF_LHS, N = {N_lhs_dist}")
        # ax1[i].plot(t, x_post_qf[i, :], label = "UKF_qf")
        ax1[i].plot(t, x_ol[i, :], label = "OL")
        ax1[i].set_ylabel(ylabels[i])
    ax1[-1].set_xlabel("Time [s]")
    #Plot measurements
    ax1[-1].plot(t, y[0,:], marker = "x", markersize = 3, linewidth = 0, label = "y")
    ax1[-1].set_ylabel(ylabels[-1])
    ax1[0].legend()        

print(f"Repeated {N} time(s). In every iteration, the number of model evaluations for computing noise statistics:\n",
      f"Q by UT: {sigmas_fx.shape[1]}\n",
      f"Q by LHS: {N_lhs_dist}\n",
      f"R by UT: {sigmas_hx.shape[1]}\n",
      f"R by LHS: {N_lhs_dist}\n")
print("Median value of cost function is\n")
for i in range(dim_x):
    print(f"{ylabels[i]}: Q-UT = {np.median(j_valappil[i]): .3f}, Q-LHS-{N_lhs_dist} = {np.median(j_valappil_lhs[i]): .3f} and Q-fixed = {np.median(j_valappil_qf[i]): .3f}")

#%% Violin plot of cost function
if N > 5: #only plot this if we have done some iterations
    fig_v, ax_v = plt.subplots(dim_x,1)
    labels_violin = ["UT", "LHS"]
    # labels_violin = ["UT", "LHS", "Fixed"]
    def set_axis_style(ax, labels):
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')
    for i in range(dim_x):
        data = np.vstack([j_valappil[i], j_valappil_lhs[i]]).T
        # data = np.vstack([j_valappil[i], j_valappil_lhs[i], j_valappil_qf[i]]).T
        ax_v[i].violinplot(data)#, j_valappil_qf])
        set_axis_style(ax_v[i], labels_violin)
        ax_v[i].set_ylabel(fr"Cost $x_{i+1}$")
        fig_v.suptitle(f"Cost function distribution for N = {N} iterations")
