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
import pathlib
import os
import scipy.linalg
import matplotlib.patches as plt_patches
import matplotlib.transforms as plt_transforms
import copy
import pickle
# Did some modification to these packages
from myFilter import UKF
from myFilter import sigma_points as ukf_sp
# from myFilter import UKF_constrained

#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import utils_falling_body as utils_fb

N_LHS_list = np.logspace(1,3,num = 20, dtype = int)
# N_LHS_list = np.array([10, 15, 20]) #for test
n_filters = 5
mean_cost = {"x1": np.zeros((n_filters, N_LHS_list.shape[0])),
             "x2": np.zeros((n_filters, N_LHS_list.shape[0])),
             "x3": np.zeros((n_filters, N_LHS_list.shape[0]))
             }
sigma_cost = {"x1": np.zeros((n_filters, N_LHS_list.shape[0])),
             "x2": np.zeros((n_filters, N_LHS_list.shape[0])),
             "x3": np.zeros((n_filters, N_LHS_list.shape[0]))
             }
for q in range(len(N_LHS_list)):
    #%% For running the sim N times
    N = 100 # number of times every simulation is repeated
    N_lhs_dist = N_LHS_list[q]
    N_mc_dist = N_LHS_list[q]
    dim_x = 3
    j_valappil = np.zeros((dim_x, N))
    j_valappil_lhs = np.zeros((dim_x, N))
    j_valappil_mc = np.zeros((dim_x, N))
    j_valappil_mcm = np.zeros((dim_x, N))
    j_valappil_qf = np.zeros((dim_x, N))
    Ni = 0
    rand_seed = 1234
    while Ni < N:
        try:
            np.random.seed(rand_seed) #to get reproducible results. rand_seed updated in every iteration
            #%% Import the distributions of the parameters for the fx equations (states)
            #Modes of the dists are used for the true system, and mean of the dists are the parameters for the UKF
            
            # utils_fb.uncertainty_venturi2()
            _, _, Q_nom, R_nom = utils_fb.get_literature_values()
            par_dist_fx, par_det_fx, fig_p_fx, ax_p_fx = utils_fb.get_param_ukf_case1(
                par_fx = True,
                std_dev_prct = 0.05, 
                plot_dist = False)
            
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
            
                # print(f"{key}_true/{key}_KF: {par_true_fx[key]/par_kf_fx[key]}")
            # par_kf_fx = par_true_fx
            #%% Import the distributions of the parameters for the hx equations (measurements)
            #Modes of the dists are used for the true system, and mean of the dists are the parameters for the UKF
            
            par_dist_hx, par_det_hx, fig_p_hx, ax_p_hx = utils_fb.get_param_ukf_case1(
                par_fx = False, #return based on par_true_hx in stead
                std_dev_prct = 0.05, 
                plot_dist = False)
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
            
                # print(f"{key}_true/{key}_KF: {par_true_hx[key]/par_kf_hx[key]}")
            # par_kf_hx = par_true_hx
            #Measurement equation also need y_repeatability
            y_rep = scipy.stats.norm(loc = 0.,
                                     scale = np.sqrt(R_nom[0])) #this is just 1D, so ok
            par_true_hx["y_rep"] = y_rep.rvs()
            par_kf_hx["y_rep"] = y_rep.mean()
            par_dist_hx["y_rep"] = y_rep
            
            #%% Define samples for LHS
            # N_lhs_dist = 10
            labels_fx = par_true_fx.copy()
            labels_fx.pop("g")
            par_lhs_fx, samples_lhs_fx, fig_lhs, ax_lhs = utils_fb.get_lhs_points(par_dist_fx,
                                                                          N_lhs_dist = N_lhs_dist,
                                                                          plot_mc_samples=False,
                                                                          labels = list(labels_fx.keys()))
            par_lhs_fx["g"] = np.ones(N_lhs_dist)*par_kf_fx["g"] #append with constant g
            par_lhs_hx, samples_lhs_hx, fig_lhs, ax_lhs = utils_fb.get_lhs_points(par_dist_hx, 
                                                                                  N_lhs_dist = N_lhs_dist, 
                                                                                  plot_mc_samples=False,
                                                                                  labels = list(par_true_hx.keys())
                                                                                  )
            #%% Define samples for MC, random sampling
            # N_mc_dist = int(1e2)
            
            par_mc_fx, fig_mc, ax_mc = utils_fb.get_mc_points(par_dist_fx, 
                                                                N_mc_dist = N_mc_dist, 
                                                                plot_mc_samples=False,
                                                                labels = labels_fx
                                                                 )
            par_mc_fx["g"] = np.ones(N_mc_dist)*par_kf_fx["g"] #append with constant g
            par_mc_hx, fig_mc, ax_mc = utils_fb.get_mc_points(par_dist_hx, 
                                                                N_mc_dist = N_mc_dist, 
                                                                plot_mc_samples=False,
                                                                labels = list(par_true_hx.keys())
                                                                 )
            #%% Define samples for MCm, random sampling
            N_mcm_dist = int(5e2)
            
            par_mcm_fx, fig_mcm, ax_mcm = utils_fb.get_mc_points(par_dist_fx, 
                                                                N_mc_dist = N_mcm_dist, 
                                                                plot_mc_samples=False,
                                                                labels = labels_fx
                                                                 )
            par_mcm_fx["g"] = np.ones(N_mcm_dist)*par_kf_fx["g"] #append with constant g
            par_mcm_hx, fig_mcm, ax_mcm = utils_fb.get_mc_points(par_dist_hx, 
                                                                N_mc_dist = N_mcm_dist, 
                                                                plot_mc_samples=False,
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
            dt_y = .5 # [s] <=> 50 ms. Measurement frequency
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
            x_post_mc = np.zeros((dim_x, dim_t))
            x_post_mcm = np.zeros((dim_x, dim_t))
            x_post_qf = np.zeros((dim_x, dim_t))
            # x_prior = np.zeros((dim_x, dim_ty))
            
            x_true[:, 0] = x0
            x_post_lhs[:, 0] = x0_kf
            x_post_mc[:, 0] = x0_kf
            x_post_mcm[:, 0] = x0_kf
            x_post[:, 0] = x0_kf
            x_post_qf[:, 0] = x0_kf
            x0_ol = copy.deepcopy(x0_kf)
            x_ol[:, 0] = x0_ol
            # x_prior[:, 0] = x0_kf
            t_span = (t[0],t[1])
            
            #%% Create noise
            w_plant = np.zeros((dim_t, dim_x))
            w_noise_kf = np.zeros(dim_x)
            
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
                                                                     par_kf_fx.copy()))
            
            hx_ukf_lhs = lambda x_in: utils_fb.hx(x_in, par_kf_hx.copy())#.reshape(-1,1)
            
            #kf is where Q adapts based on UT of parametric uncertainty
            kf_lhs = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                               dim_z = dim_y, 
                                                dt = 100, 
                                                hx = hx_ukf_lhs, 
                                                fx = fx_ukf_lhs,
                                                points = points_lhs)
            kf_lhs.x = x_post_lhs[:, 0]
            kf_lhs.P = copy.deepcopy(P0)
            kf_lhs.Q = Q_nom #to be updated in a loop
            kf_lhs.R = R_nom #to be updated in a loop
            
            #%% Define UKF with adaptive Q, R from MC with random sampling
            points_mc = ukf_sp.JulierSigmaPoints(dim_x,
                                              kappa = 3-dim_x)
            fx_ukf_mc = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                         t_span, 
                                                         x,
                                                         args_ode = (w_noise_kf,
                                                                     par_kf_fx.copy()))
            
            hx_ukf_mc = lambda x_in: utils_fb.hx(x_in, par_kf_hx.copy())#.reshape(-1,1)
            
            #kf is where Q adapts based on UT of parametric uncertainty
            kf_mc = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                              dim_z = dim_y, 
                                              dt = 100, 
                                              hx = hx_ukf_mc, 
                                              fx = fx_ukf_mc,
                                                points = points_mc)
            kf_mc.x = x_post_mc[:, 0]
            kf_mc.P = copy.deepcopy(P0)
            kf_mc.Q = Q_nom #to be updated in a loop
            kf_mc.R = R_nom #to be updated in a loop
            
            #%% Define UKF with adaptive Q, R from MCm with random sampling and mode adjustment
            points_mcm = ukf_sp.JulierSigmaPoints(dim_x,
                                              kappa = 3-dim_x)
            fx_ukf_mcm = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                         t_span, 
                                                         x,
                                                         args_ode = (w_noise_kf,
                                                                     par_kf_fx.copy()))
            
            hx_ukf_mcm = lambda x_in: utils_fb.hx(x_in, par_kf_hx.copy())#.reshape(-1,1)
            
            #kf is where Q adapts based on UT of parametric uncertainty
            kf_mcm = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                              dim_z = dim_y, 
                                              dt = 100, 
                                              hx = hx_ukf_mcm, 
                                              fx = fx_ukf_mcm,
                                                points = points_mcm)
            kf_mcm.x = x_post_mcm[:, 0]
            kf_mcm.P = copy.deepcopy(P0)
            kf_mcm.Q = Q_nom #to be updated in a loop
            kf_mcm.R = R_nom #to be updated in a loop
            
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
                                              dt = None, #not used
                                              hx = hx_ukf_qf, 
                                              fx = fx_ukf_qf, 
                                              points = points_qf)
            kf_qf.x = x_post[:, 0]
            kf_qf.P = copy.deepcopy(P0)
            kf_qf.Q = Q_nom
            kf_qf.R = R_nom*100
            
            
            #%% Get parametric uncertainty of fx by GenUT
            sigmas_fx, w_fx = utils_fb.get_sigmapoints_and_weights(par_dist_fx)
            list_dist_fx_keys = list(par_dist_fx.keys()) # list of parameters with distribution
            fx_gen_Q = lambda si: utils_fb.fx_for_UT_gen_Q(si, 
                                                           list_dist_fx_keys, 
                                                           t_span, 
                                                           x0, 
                                                           par_kf_fx.copy(),
                                                           w_noise_kf)
            
            #%% Get parametric uncertainty of hx by GenUT
            sigmas_hx, w_hx = utils_fb.get_sigmapoints_and_weights(par_dist_hx)
            list_dist_hx_keys = list(par_dist_hx.keys()) # list of parameters with distribution
            hx_gen_R = lambda si: utils_fb.hx_for_UT_gen_R(si, 
                                                            list_dist_hx_keys, 
                                                            x0, 
                                                            par_kf_hx.copy())
            # print("---par_fx before sim---\n",
            #       f"true: {par_true_fx}\n",
            #       f"kf: {par_kf_fx}\n")
            
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
                
                
                
                x_nom_ut = utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                               t_span, 
                                               x_post[:, i-1], 
                                               args_ode = (w_noise_kf, par_kf_fx))
                fx_gen_Q = lambda si: (utils_fb.fx_for_UT_gen_Q(si, 
                                                               list_dist_fx_keys, 
                                                               t_span, 
                                                               x_post[:, i-1], 
                                                               par_kf_fx.copy(),
                                                               w_noise_kf)
                                       - x_nom_ut
                                       )
                #Adaptive Q by UT
                w_mean_ut, Q_ut = ut.unscented_transformation(sigmas_fx, w_fx, fx = fx_gen_Q)
                
                #Adaptive Q by LHS
                w_mean_lhs, Q_lhs = utils_fb.get_wmean_Q_from_mc(par_lhs_fx.copy(), #same function as mc
                                                x_post_lhs[:, i-1], 
                                                t_span, 
                                                w_noise_kf,
                                                par_kf_fx)
                
                #Adaptive Q by MC random
                w_mean_mc, Q_mc = utils_fb.get_wmean_Q_from_mc(par_mc_fx.copy(), #get_wmeanXXX or get_wmodeXXX
                                                x_post_mc[:, i-1], 
                                                t_span, 
                                                w_noise_kf,
                                                par_kf_fx)
                
                #Adaptive Q by MC random with mode adjustment
                w_mode_mcm, Q_mcm = utils_fb.get_wmode_Q_from_mc(par_mcm_fx.copy(), #get_wmeanXXX or get_wmodeXXX
                                                x_post_mcm[:, i-1], 
                                                t_span, 
                                                w_noise_kf,
                                                par_kf_fx)
                
                w_mode_mcm[-1] = w_mean_ut[-1] #important to do this for the mode adjustemnt! This is since the mode is found "graphically" by a histogram. The last state is constant ==> it is just one bin.  The middle point of that bin is a very bad value for w_mode_mcm and it makes the filter to diverge. Check the code block "#%% Check if wk is normally distributed" for a visualization of this. If this is not done, the MC filter will diverge.
                
                #Check that we don't have zeros on the diagonals
                for di in range(dim_x):
                    if (Q_ut[di,di] == 0):
                        Q_ut[di,di] = 1e-8 
                    if (Q_lhs[di,di] == 0):
                        Q_lhs[di,di] = 1e-8 
                    if (Q_mc[di,di] == 0):
                        Q_mc[di,di] = 1e-8
                    if (Q_mcm[di,di] == 0):
                        Q_mcm[di,di] = 1e-8
                kf.Q = Q_ut
                kf_lhs.Q = Q_lhs
                kf_mc.Q = Q_mc
                kf_mcm.Q = Q_mcm
                
                #Prediction step of kalman filter. Make the functions for each UKF
                fx_ukf = lambda x, dt_kf: (utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                              t_span, 
                                                              x,
                                                              args_ode = (w_noise_kf,
                                                                          par_kf_fx)
                                                              )
                                           # + w_mean_ut #this line can be commented out
                                           )
                fx_ukf_lhs = lambda x, dt_kf: (utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                              t_span, 
                                                              x,
                                                              args_ode = (w_noise_kf,
                                                                          par_kf_fx)
                                                              )
                                                # + w_mean_lhs
                                               )
                fx_ukf_mc = lambda x, dt_kf: (utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                              t_span, 
                                                              x,
                                                              args_ode = (w_noise_kf,
                                                                          par_kf_fx)
                                                              )
                                               # + w_mean_mc # This line can be commented out
                                              )
                fx_ukf_mcm = lambda x, dt_kf: (utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                              t_span, 
                                                              x,
                                                              args_ode = (w_noise_kf,
                                                                          par_kf_fx)
                                                              )
                                                + w_mode_mcm # This line can be commented out
                                              )
                fx_ukf_qf = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                              t_span, 
                                                              x,
                                                              args_ode = (w_noise_kf,
                                                                          par_kf_fx)
                                                              )
                
                #Prediction step of each UKF
                kf.predict(fx = fx_ukf)
                kf_lhs.predict(fx = fx_ukf_lhs)
                kf_mc.predict(fx = fx_ukf_mc)
                kf_mcm.predict(fx = fx_ukf_mcm)
                kf_qf.predict(fx = fx_ukf_qf)
                # print(f"pred: {kf.x}")
                
                #Make a new measurement
                par_true_hx["y_rep"] = y_rep.rvs() #draw a new sample from noise statistics
                y[:, i] = utils_fb.hx(x_true[:, i], par_true_hx) #add the measurement
                #%% Adaptive R
                #Adaptive R by UT
                x_prior = copy.deepcopy(kf.x_prior)
                y_nom = utils_fb.hx(x_prior, par_kf_hx)
                hx_gen_R = lambda si: (utils_fb.hx_for_UT_gen_R(si, 
                                                                       list_dist_hx_keys, 
                                                                       x_prior, 
                                                                       par_kf_hx.copy())
                                       - y_nom
                                       )
                v_mean_ut, R_ut = ut.unscented_transformation(sigmas_hx, 
                                                            w_hx, 
                                                            fx = hx_gen_R)
                kf.R = R_ut
                
                #Adaptive R by LHS
                x_prior_lhs = copy.deepcopy(kf_lhs.x_prior)
                v_mean_lhs, R_lhs = utils_fb.get_vmean_R_from_mc(par_lhs_hx.copy(), 
                                                                x_prior_lhs, 
                                                                dim_y,
                                                                par_kf_hx)
                kf_lhs.R = R_lhs
                
                #Adaptive R by MC
                x_prior_mc = copy.deepcopy(kf_mc.x_prior)
                v_mean_mc, R_mc = utils_fb.get_vmean_R_from_mc(par_mc_hx.copy(), 
                                                          x_prior_mc, 
                                                          dim_y,
                                                          par_kf_hx)
                kf_mc.R = np.array([[R_mc]]) #it is 1D array
                
                #Adaptive R by MCm (mode adjusted)
                x_prior_mcm = copy.deepcopy(kf_mc.x_prior)
                v_mode_mcm, R_mcm = utils_fb.get_vmode_R_from_mc(par_mcm_hx.copy(), 
                                                          x_prior_mcm, 
                                                          dim_y,
                                                          par_kf_hx)
                kf_mcm.R = np.array([[R_mcm]]) #it is 1D array
                
                # hx_ukf2 = lambda x_in: hx_ukf(x_in) + v_mean_ut
                # hx_ukf_lhs2 = lambda x_in: hx_ukf_lhs(x_in) + v_mean_lhs
                # hx_ukf_mc2 = lambda x_in: hx_ukf_mc(x_in) + v_mean_mc
                hx_ukf_mcm2 = lambda x_in: hx_ukf_mcm(x_in) + v_mode_mcm
                
                #Correction step of UKF with v_mean
                # kf.update(y[:, i], hx = hx_ukf2)
                # kf_lhs.update(y[:, i], hx = hx_ukf_lhs2)
                # kf_mc.update(y[:, i], hx = hx_ukf_mc2)
                kf_mcm.update(y[:, i], hx = hx_ukf_mcm2)
                # kf_qf.update(y[:, i], hx = hx_ukf_qf)
                
                #Correction step of UKF wihout v_mean
                kf.update(y[:, i], hx = hx_ukf)
                kf_lhs.update(y[:, i], hx = hx_ukf_lhs)
                kf_mc.update(y[:, i], hx = hx_ukf_mc)
                kf_qf.update(y[:, i], hx = hx_ukf_qf)
                
                # Save the estimates
                x_post[:, i] = kf.x
                x_post_lhs[:, i] = kf_lhs.x
                x_post_mc[:, i] = kf_mc.x
                x_post_mcm[:, i] = kf_mcm.x
                x_post_qf[:, i] = kf_qf.x
                
            
            y[:, 0] = np.nan #the 1st measurement is not real, just for programming convenience
            
            
            
            #%% Compute performnance index
            j_valappil[:, Ni] = utils_fb.compute_performance_index_valappil(x_post, 
                                                                         x_ol, 
                                                                         x_true)
            j_valappil_lhs[:, Ni] = utils_fb.compute_performance_index_valappil(x_post_lhs, 
                                                                             x_ol, 
                                                                             x_true)
            j_valappil_mc[:, Ni] = utils_fb.compute_performance_index_valappil(x_post_mc, 
                                                                             x_ol, 
                                                                             x_true)
            j_valappil_mcm[:, Ni] = utils_fb.compute_performance_index_valappil(x_post_mcm, 
                                                                             x_ol, 
                                                                             x_true)
            j_valappil_qf[:, Ni] = utils_fb.compute_performance_index_valappil(x_post_qf, 
                                                                            x_ol, 
             x_true)
            
            Ni += 1
            rand_seed += 1
            if (Ni%20 == 0): #print every 5th iteration                                                               
                print(f"End of subiteration {Ni}/{N} in main iteration {q+1}/{len(N_LHS_list)}. N_lhs_dist = {N_lhs_dist}")
        except BaseException as e:
            # print(e)
            # raise e
            print("Error occured - skipped this np.random.seed")
            rand_seed += 1 #sometimes we get an error. Just skip that seed until the reason why is figured out
            continue
    #Compute mean cost and sigma cost
    for i in range(dim_x):
        cost = np.vstack([j_valappil[i], j_valappil_lhs[i], j_valappil_mc[i],
                          j_valappil_mcm[i], j_valappil_qf[i]]).T
        mean_cost["x"+str(i+1)][:, q] = cost.mean(axis = 0)
        sigma_cost["x"+str(i+1)][:, q] = cost.std(axis = 0)
# print("---par_fx after sim---\n",
#               f"true: {par_true_fx}\n",
#               f"kf: {par_kf_fx}\n")
#%% Plot
plot_it = True
if plot_it:
    # ylabels = [r"$x_1 [ft]$", r"$x_2 [ft/s]$", r"$x_3 [ft^3$/(lb-$s^2)]$", "$y [ft]$"]#
    ylabels = [r"$x_1$ [ft]", r"$x_2$ [ft/s]", r"$x_3$ [*]", "$y$ [ft]"]#
        
    fig1, ax1 = plt.subplots(dim_x + 1, 1, sharex = True)
    for i in range(dim_x): #plot true states and ukf's estimates
        ax1[i].plot(t, x_true[i, :], label = "True")
        ax1[i].plot([np.nan, np.nan], [np.nan, np.nan], color='w', alpha=0, label=' ')
        ax1[i].plot(t, x_post[i, :], label = r"GenUT")
        # ax1[i].plot(t, x_post_lhs[i, :], label = r"LHS")
        # ax1[i].plot(t, x_post_mc[i, :], label = r"MC")
        ax1[i].plot(t, x_post_mcm[i, :], label = r"MCm")
        ax1[i].plot(t, x_post_qf[i, :], label = r"Fixed")
        
        # ax1[i].plot(t, x_post[i, :], label = r"$Q_{UT}$")
        # ax1[i].plot(t, x_post_lhs[i, :], label = r"$Q_{LHS}$")
        # ax1[i].plot(t, x_post_mc[i, :], label = r"$Q_{MC}$")
        # ax1[i].plot(t, x_post_mcm[i, :], label = r"$Q_{MCm}$")
        # ax1[i].plot(t, x_post_qf[i, :], label = r"$Q_{fixed}$")
        
        # ax1[i].plot(t, x_post[i, :], label = r"UKF, $Q_{UT}$")
        # ax1[i].plot(t, x_post_lhs[i, :], label = r"UKF, $Q_{LHS}$")
        # ax1[i].plot(t, x_post_qf[i, :], label = r"UKF, $Q_{fixed}$")
        ax1[i].plot(t, x_ol[i, :], label = "OL")
        ax1[i].set_ylabel(ylabels[i])
    ax1[-1].set_xlabel("Time [s]")
    #Plot measurements
    ax1[-1].plot(t, y[0,:], marker = "x", markersize = 3, linewidth = 0, label = "y")
    ax1[-1].set_ylabel(ylabels[-1])
    # ax1[0].legend()        
    ax1[0].legend(ncol = 3,
                  frameon = False)        

print(f"Repeated {N} time(s). In every iteration, the number of model evaluations for computing noise statistics:\n",
      f"Q by UT: {sigmas_fx.shape[1]}\n",
      f"Q by LHS: {N_lhs_dist}\n",
      f"Q by MC: {N_mc_dist}\n",
      f"Q by MCm: {N_mcm_dist}\n",
      f"R by UT: {sigmas_hx.shape[1]}\n",
      f"R by LHS: {N_lhs_dist}\n",
      f"R by MC: {N_mc_dist}\n",
      f"R by MCm: {N_mcm_dist}\n")
print("Median value of cost function is\n")
for i in range(dim_x):
    print(f"{ylabels[i]}: Q-UT = {np.median(j_valappil[i]): .3f}, Q-LHS-{N_lhs_dist} = {np.median(j_valappil_lhs[i]): .3f}, Q-MC-{N_mc_dist} = {np.median(j_valappil_mc[i]): .3f}, Q-MCm-{N_mcm_dist} = {np.median(j_valappil_mcm[i]): .3f} and Q-fixed = {np.median(j_valappil_qf[i]): .3f}")

#%% Violin plot of cost function
if N >= 5: #only plot this if we have done some iterations
    fig_v, ax_v = plt.subplots(dim_x,1)
    # labels_violin = ["UT", "LHS"]
    labels_violin = ["GenUT", "LHS", "MC", "MCm", "Fixed"]
    def set_axis_style(ax, labels):
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        # ax.set_xlabel(r'Method for tuning $Q_k, R_k$')
    for i in range(dim_x):
        # data = np.vstack([j_valappil[i], j_valappil_lhs[i]]).T
        data = np.vstack([j_valappil[i], j_valappil_lhs[i], j_valappil_mc[i], j_valappil_mcm[i], j_valappil_qf[i]]).T
        print("---cost of x_{i}---\n",
              f"mean = {data.mean(axis = 0)}\n",
              f"std = {data.std(axis = 0)}")
        ax_v[i].violinplot(data)#, j_valappil_qf])
        set_axis_style(ax_v[i], labels_violin)
        ax_v[i].set_ylabel(fr"Cost $x_{i+1}$")
    ax_v[-1].set_xlabel(r'Method for tuning $Q_k, R_k$')
        # fig_v.suptitle(f"Cost function distribution for N = {N} iterations")

#%% Plot of mean and variance of cost function as a function of N_LHS

if len(N_LHS_list) >= 2:
    # fig_n_m, ax_n_m = plt.subplots(dim_x,1)
    # # ax_n_m[0].set_xlabel(r"N_{LHS}")
    # # ax_n_m[0].set_ylabel(r"\mu_{cost}")
    # labels_mu = [r"$\mu_{cost}^{GenUT}$", "$\mu_{cost}^{LHS}$", 
    #              "$\mu_{cost}^{MC}$", "$\mu_{cost}^{MCm}$", "$\mu_{cost}^{fixed}$"]
    
    # for i in range(dim_x):
    #     for j in range(len(labels_mu)):
    #         ax_n_m[i].plot(N_LHS_list, 
    #                        mean_cost["x"+str(i+1)][j, :], 
    #                        label = labels_mu[j], marker = "x")
    #     ax_n_m[i].set_xlabel(r"$N_{LHS}$")
    #     k = i+1
    #     ax_n_m[i].set_ylabel(rf"$x_{k}$")
    # ax_n_m[0].legend(ncol = 3,
    #                  frameon = False)
    
    
    fig_n_s, ax_n_s = plt.subplots(dim_x,1, sharex = True)
    # ax_n_s[0].set_xlabel(r"N_{LHS}")
    # ax_n_s[0].set_ylabel(r"\sigma_{cost}")
    # labels_sigma = [r"$\sigma_{cost}^{GenUT}$", "$\sigma_{cost}^{LHS}$",
    #                 "$\sigma_{cost}^{MC}$", "$\sigma_{cost}^{MCm}$", "$\sigma_{cost}^{fixed}$"]
    # labels_sigma = [r"$\sigma_{cost}^{GenUT}$", "$\sigma_{cost}^{LHS}$",
    #                 "$\sigma_{cost}^{MC}$", "$\sigma_{cost}^{MCm}$", "$\sigma_{cost}^{fixed}$"]
    labels_sigma = [r"GenUT", "LHS", "MC", "MCm", "Fixed"]
    ylabels = [r"$\sigma^{cost}_{x_1}$", r"$\sigma^{cost}_{x_2}$", r"$\sigma^{cost}_{x_3}$"]
    for i in range(dim_x):
        for j in range(len(labels_sigma)):
            if j >= 3:
                continue
            ax_n_s[i].plot(N_LHS_list, 
                           sigma_cost["x"+str(i+1)][j, :], 
                           label = labels_sigma[j], marker = "x")
        
        k = i+1
        # ax_n_s[i].set_ylabel(rf"$x_{k}$")
        # ylabel = r"$\sigma^{cost}_$" + f"x_{k}"
        ax_n_s[i].set_ylabel(ylabels[i])
    ax_n_s[i].set_xlabel(r"$N_{LHS}$ and $N_{MC}$")
    ax_n_s[0].legend(ncol = 3,
                     frameon = False)

#%% Save estimates    
with open('mean_cost_5filters.pickle', 'wb') as handle:
    pickle.dump(mean_cost, handle, protocol=pickle.HIGHEST_PROTOCOL)   
with open('sigma_cost_5filters.pickle', 'wb') as handle:
    pickle.dump(sigma_cost, handle, protocol=pickle.HIGHEST_PROTOCOL)
np.save("N_lhs_and_mc_list_5filters.npy", N_LHS_list)


#%% Check if wk is normally distributed
if False:
    N_mc = int(1e5)
    nbins = 50
    x_prior = x_post[:, 0]
    t_span = (t[0], t[1])
    
    par_mc_fx, fig_mc, ax_mc = utils_fb.get_mc_points(par_dist_fx, 
                                                            N_mc_dist = N_mc, 
                                                            plot_mc_samples=False,
                                                            labels = labels_fx
                                                             )
    par_mc_fx["g"] = np.ones(N_mc)*par_kf_fx["g"] #append with constant g
    
    x_nom = utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                           t_span, 
                                           x_prior, 
                                           args_ode = (w_noise_kf, par_kf_fx))
    fx_gen_Q = lambda si: (utils_fb.fx_for_UT_gen_Q(si, 
                                                   list_dist_fx_keys, 
                                                   t_span, 
                                                   x_prior, 
                                                   par_kf_fx.copy(),
                                                   w_noise_kf)
                           - x_nom
                           )
    #Adaptive Q by UT
    w_mean_ut, Q_ut = ut.unscented_transformation(sigmas_fx, w_fx, fx = fx_gen_Q)
    
    #Adaptive Q by LHS
    w_mean_lhs, Q_lhs = utils_fb.get_wmean_Q_from_mc(par_lhs_fx.copy(), #same function as mc
                                    x_prior, 
                                    t_span, 
                                    w_noise_kf,
                                    par_kf_fx)
    
    #Adaptive Q by MC random
    w_mean_mc, Q_mc = utils_fb.get_wmean_Q_from_mc(par_mc_fx.copy(), 
                                    x_prior, 
                                    t_span, 
                                    w_noise_kf,
                                    par_kf_fx)
    w_mode_mc, Q_mc = utils_fb.get_wmode_Q_from_mc(par_mc_fx.copy(), 
                                    x_prior, 
                                    t_span, 
                                    w_noise_kf,
                                    par_kf_fx,
                                    nbins = nbins)
    
    w_ukf = utils_fb.get_w_realizations_from_mc(par_mc_fx.copy(), 
                                                x_prior, 
                                                t_span, 
                                                w_noise_kf, 
                                                par_kf_fx)
    Q_ut = np.diag(Q_ut)
    Q_lhs = np.diag(Q_lhs)
    Q_mc = np.diag(Q_mc)

    
    x_plant_para = utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                        t_span, 
                                        x_prior, 
                                        args_ode = (w_noise_kf, par_true_fx))
    w_plant_para = x_plant_para - x_nom
    
    fig_w, ax_w = plt.subplots(dim_x,1)
    for i in range(dim_x):
        axi = ax_w[i]
        
        
        (n, bins, patches) = axi.hist(w_ukf[i, :].T, bins = nbins, alpha =.2, density = False, label = r"$w_{UKF}$")
        l_ut = axi.scatter(w_mean_ut[i], [0],label = r"$w_{mean}^{UT} \pm \sqrt{Q^{UT}}$")
        l_lhs = axi.scatter(w_mean_lhs[i], [0], label = "$w_{mean}^{LHS} \pm \sqrt{Q^{LHS}}$")
        l_mc = axi.scatter(w_mean_mc[i], [0], label = "$w_{mean}^{MC} \pm \sqrt{Q^{MC}}$")
        axi.scatter(w_plant_para[i], [0], label = "$w_{plant}$")
        axi.scatter(w_mode_mc[i], [0], label = "$w_{mode}$")
        
        ylim = axi.get_ylim()
        #plot limits mean +/- std dev for UT,
        kwargs_ut = {"color": l_ut.get_facecolor(),
                     "linestyle": "solid"}
        axi.plot(np.array([w_mean_ut[i] - np.sqrt(Q_ut[i]), w_mean_ut[i] - np.sqrt(Q_ut[i])]).flatten(),
                  list(ylim),
                  **kwargs_ut)
        axi.plot(np.array([w_mean_ut[i] + np.sqrt(Q_ut[i]), w_mean_ut[i] + np.sqrt(Q_ut[i])]).flatten(),
                  list(ylim),
                  **kwargs_ut)
        #plot limits mean +/- std dev for LHS,
        kwargs_lhs = {"color": l_lhs.get_facecolor(),
                     "linestyle": "dashed"}
        axi.plot(np.array([w_mean_lhs[i] - np.sqrt(Q_lhs[i]), w_mean_lhs[i] - np.sqrt(Q_lhs[i])]).flatten(),
                  list(ylim),
                  **kwargs_lhs)
        axi.plot(np.array([w_mean_lhs[i] + np.sqrt(Q_lhs[i]), w_mean_lhs[i] + np.sqrt(Q_lhs[i])]).flatten(),
                  list(ylim),
                  **kwargs_lhs)
        #plot limits mean +/- std dev for MC,
        kwargs_mc = {"color": l_mc.get_facecolor(),
                     "linestyle": "dashed"}
        axi.plot(np.array([w_mean_mc[i] - np.sqrt(Q_mc[i]), w_mean_mc[i] - np.sqrt(Q_mc[i])]).flatten(),
                  list(ylim),
                  **kwargs_mc)
        axi.plot(np.array([w_mean_mc[i] + np.sqrt(Q_mc[i]), w_mean_mc[i] + np.sqrt(Q_mc[i])]).flatten(),
                  list(ylim),
                  **kwargs_mc)
        # axi.scatter(y_plant_para, [10], label = "Plant parameter")
        
        #find the mode by maximizing the histogram values
        hist, bin_edges = np.histogram(w_ukf, bins = nbins)
        idx = np.argmax(hist) #returns the index where there are most samples in the bin
        [low_lim, h_lim] = [bin_edges[idx], bin_edges[idx+1]]
        axi.scatter([low_lim, h_lim], [hist[idx], hist[idx]], label = "Mode range")
        
        axi.set_ylim(ylim)
        axi.set_ylabel(f"hist values for w_{i}")
    ax_w[0].legend(ncol = 7,
                   frameon = False)
    axi.set_xlabel(r"$w^{UKF}$ realizations")
#%% Check if vk is normally distributed by code2
if False:
    N_mc = int(1e5)
    nbins = 30
    x_prior = x_post[:, 0]
    
    par_mc_hx, fig_mc, ax_mc = utils_fb.get_mc_points(par_dist_hx, 
                                                    N_mc_dist = N_mc, 
                                                    plot_mc_samples=False,
                                                    labels = list(par_true_hx.keys())
                                                                 )
    # y_rep = scipy.stats.norm(loc = 0.,
    #                              scale = np.sqrt(R_nom[0])) #this is just 1D, so ok
    # par_true_hx["y_rep"] = y_rep.rvs()
    # par_kf_hx["y_rep"] = y_rep.mean()
    # par_dist_hx["y_rep"] = y_rep
    
    # par_mc_hx, fig_mc, ax_mc = utils_fb.get_mc_points(par_dist_hx, 
    #                                                         N_mc_dist = N_mc, 
    #                                                         plot_mc_samples=False,
    #                                                         labels = labels_hx
    #                                                          )
    
    y_nom = utils_fb.hx(x_prior, par_kf_hx)
    hx_gen_R = lambda si: (utils_fb.hx_for_UT_gen_R(si, 
                                                           list_dist_hx_keys, 
                                                           x_prior, 
                                                           par_kf_hx.copy())
                           - y_nom
                           )
    v_mean_ut, R_ut = ut.unscented_transformation(sigmas_hx, 
                                                w_hx, 
                                                fx = hx_gen_R)
    
    #Adaptive Q by LHS
    v_mean_lhs, R_lhs = utils_fb.get_vmean_R_from_mc(par_lhs_hx.copy(), #same function as mc
                                    x_prior, 
                                    dim_y,
                                    par_kf_hx)
    
    #Adaptive Q by MC random
    v_mean_mc, R_mc = utils_fb.get_vmean_R_from_mc(par_mc_hx.copy(), 
                                    x_prior,
                                    dim_y,
                                    par_kf_hx)
    v_mode_mc, R_mc = utils_fb.get_vmode_R_from_mc(par_mc_hx.copy(), 
                                    x_prior,
                                    dim_y,
                                    par_kf_hx,
                                    nbins = nbins)
    
    v_ukf = utils_fb.get_v_realizations_from_mc(par_mc_hx.copy(), 
                                                x_prior, 
                                                dim_y,
                                                par_kf_hx)
    # R_ut = np.diag(R_ut)
    # R_lhs = np.diag(R_lhs)
    # R_mc = np.diag(R_mc)

    
    y_plant_para = utils_fb.hx(x_prior, par_true_hx)
    v_plant_para = y_plant_para - y_nom
    
    fig_v, ax_v = plt.subplots(dim_y,1)
    
    
    
    (n, bins, patches) = ax_v.hist(v_ukf.T, bins = nbins, alpha =.2, density = False, label = r"$v_{UKF}$")
    l_ut = ax_v.scatter(v_mean_ut, [0],label = r"$v_{mean}^{UT} \pm \sqrt{Q^{UT}}$")
    l_lhs = ax_v.scatter(v_mean_lhs, [0], label = "$v_{mean}^{LHS} \pm \sqrt{Q^{LHS}}$")
    l_mc = ax_v.scatter(v_mean_mc, [0], label = "$v_{mean}^{MC} \pm \sqrt{Q^{MC}}$")
    ax_v.scatter(v_plant_para, [0], label = "$v_{plant}$")
    ax_v.scatter(v_mode_mc, [0], label = "$v_{mode}$")
    
    ylim = ax_v.get_ylim()
    #plot limits mean +/- std dev for UT,
    kwargs_ut = {"color": l_ut.get_facecolor(),
                 "linestyle": "solid"}
    ax_v.plot(np.array([v_mean_ut - np.sqrt(R_ut), v_mean_ut - np.sqrt(R_ut)]).flatten(),
              list(ylim),
              **kwargs_ut)
    ax_v.plot(np.array([v_mean_ut + np.sqrt(R_ut), v_mean_ut + np.sqrt(R_ut)]).flatten(),
              list(ylim),
              **kwargs_ut)
    #plot limits mean +/- std dev for LHS,
    kwargs_lhs = {"color": l_lhs.get_facecolor(),
                 "linestyle": "dashed"}
    ax_v.plot(np.array([v_mean_lhs - np.sqrt(R_lhs), v_mean_lhs - np.sqrt(R_lhs)]).flatten(),
              list(ylim),
              **kwargs_lhs)
    ax_v.plot(np.array([v_mean_lhs + np.sqrt(R_lhs), v_mean_lhs + np.sqrt(R_lhs)]).flatten(),
              list(ylim),
              **kwargs_lhs)
    #plot limits mean +/- std dev for MC,
    kwargs_mc = {"color": l_mc.get_facecolor(),
                 "linestyle": "dashed"}
    ax_v.plot(np.array([v_mean_mc - np.sqrt(R_mc), v_mean_mc - np.sqrt(R_mc)]).flatten(),
              list(ylim),
              **kwargs_mc)
    ax_v.plot(np.array([v_mean_mc + np.sqrt(R_mc), v_mean_mc + np.sqrt(R_mc)]).flatten(),
              list(ylim),
              **kwargs_mc)
    # ax_v.scatter(y_plant_para, [10], label = "Plant parameter")
    
    #find the mode by maximizing the histogram values
    hist, bin_edges = np.histogram(v_ukf, bins = nbins)
    idx = np.argmax(hist) #returns the index where there are most samples in the bin
    [low_lim, h_lim] = [bin_edges[idx], bin_edges[idx+1]]
    ax_v.scatter([low_lim, h_lim], [hist[idx], hist[idx]], label = "Mode range")
    
    ax_v.set_ylim(ylim)
    ax_v.set_ylabel(f"hist values for v_{i}")
    ax_v.legend(ncol = 2,
                   frameon = False)
    ax_v.set_xlabel(r"$v^{UKF}$ realizations")
