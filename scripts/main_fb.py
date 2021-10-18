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

#%% Import the distributions of the parameters. 
#Modes of the dists are used for the true system, and mean of the dists are the parameters for the UKF

# utils_ip.uncertainty_venturi()
par_true, Q_nom, R_nom = utils_fb.get_literature_values()
par_dist, par_det, fig_p, ax_p = utils_fb.get_param_ukf_case1(
    std_dev_prct = 0.05, 
    plot_dist = False)

mode_dist = np.zeros(len(par_dist)) #for the mode of the distributions
par_true = par_det.copy()
par_kf = par_det.copy()
del par_det
for key, dist in par_dist.items():
    #Need to find the mode numericlly
    mode = scipy.optimize.minimize(lambda theta: -dist.pdf(theta),
                                   dist.mean(), #use mean as theta0
                                   tol = 1e-10) 
    par_true[key] = mode.x #true system uses the mode values
    par_kf[key] = dist.mean() #the ukf uses mean values reported in the literaure

    print(f"{key}_true/{key}_KF: {par_true[key]/par_kf[key]}")
par_kf = copy.deepcopy(par_true)
#%% Define dimensions and initialize arrays
sigmas, w = utils_fb.get_sigmapoints_and_weights(par_dist)

x0 = utils_fb.get_x0_literature()
dim_x = x0.shape[0]
dt_y = .5 # [s] <=> 5 ms. Measurement frequency
dt_int = 1e-3 #[s] - discretization time for integration

t_end = 30
t_y = np.linspace(0, t_end, int(t_end/dt_y))
# t = np.linspace(t_y[0],t_y[0+1], int(dt_y/dt), endpoint = True)

dim_ty = t_y.shape[0]
y0 = utils_fb.hx(x0, 0., par_true)
dim_y = y0.shape[0]
y = np.zeros((dim_y, dim_ty))
y[:, 0] = y0

x_true = [[] for _ in range(dim_ty-1)] #make a list of list
t = [[] for _ in range(dim_ty-1)] #make a list of list
x_post = np.zeros((dim_x, dim_ty))
x_post[:, 0] = x0
x_prior = np.zeros((dim_x, dim_ty))
x_prior[:, 0] = x0
t_span = (t_y[0],t_y[1])

#%% Create noise
v = np.random.normal(loc = 0, scale = np.sqrt(R_nom), size = dim_ty) #white noise
w_plant = np.random.multivariate_normal(np.zeros((dim_x,)), 
                                        Q_nom, 
                                        size = dim_ty)
w_kf = np.zeros(dim_x)
v_kf = np.zeros(dim_y)

#%% Define UKF - it uses Julier's sigma points
points = ukf_sp.JulierSigmaPoints(dim_x,
                                  kappa = 0)#3-dim_x)
fx_ukf = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                             t_span, 
                                             x,
                                             args_ode = (w_kf,
                                                         par_kf))

hx_ukf = lambda x_in: utils_fb.hx(x_in, v_kf, par_kf)#.reshape(-1,1)

kf = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                               dim_z = dim_y, 
                                dt = 100, 
                                hx = hx_ukf, 
                                fx = fx_ukf,
                                points = points)
kf.x = x_post[:, 0]
kf.P = np.diag([1e6,#[ft^2], altitute
                4e6, # [ft^2], horizontal range
                1e6 # [?] ballistic coefficient
                ])
# kf.P = np.diag([1e6,#[ft^2], altitute
#                 4e6, # [ft^2], horizontal range
#                 1e1 # [?] ballistic coefficient
#                 ])
kf.Q = Q_nom
kf.R = R_nom


#%% Get parametric uncertainty by GenUT
sigmas, w = utils_fb.get_sigmapoints_and_weights(par_dist)
list_dist_keys = list(par_dist.keys()) # list of parameters with distribution
fx_gen_Q = lambda si: utils_fb.fx_for_UT_gen_Q(si, 
                                               list_dist_keys, 
                                               t_span, 
                                               x0, 
                                               par_kf,
                                               w_kf)
mean_ut, Q_ut = ut.unscented_transformation(sigmas, 
                                            w, 
                                            fx = fx_gen_Q)

#%% Simulate the plant and UKF
for i in range(1,dim_ty):
    t_span = (t_y[i-1],t_y[i])
    w_plant_i = w_plant[i, :]
    res = scipy.integrate.solve_ivp(utils_fb.ode_model_plant, 
                                    t_span,#(t_y[i-1],t_y[i]), 
                                    x0, 
                                    # rtol = 1e-10,
                                    # atol = 1e-13
                                    args = (w_plant_i, par_true)
                                    )
    t[i-1] = res.t #add integration time to the list
    x_true[i-1] = res.y #add the interval to the full list
    
    #Prediction step of kalman filter
    
    fx_ukf = lambda x, dt_kf: utils_fb.fx_ukf_ode(utils_fb.ode_model_plant, 
                                                  t_span, 
                                                  x,
                                                  args_ode = (w_kf,
                                                              par_true)
                                                  )
    fx_gen_Q = lambda si: utils_fb.fx_for_UT_gen_Q(si, 
                                                   list_dist_keys, 
                                                   t_span, 
                                                   x_post[:, i-1], 
                                                   par_kf,
                                                   w_kf)
    mean_ut, Q_ut = ut.unscented_transformation(sigmas, w, fx = fx_gen_Q)
    # kf.Q = Q_ut
    
    kf.predict(fx = fx_ukf)
    # print(f"pred: {kf.x}")
    #Get measurement
    x0 = res.y[:, -1] #starting point for next integration interval
    y[:, i] = utils_fb.hx(x0, v[i], par_true) #add the measurement

    #Correction step of UKF
    
    kf.update(y[:, i], hx = hx_ukf)
    # print(f"post: {kf.x}")
    # kf.update(np.array([y[i]]))
    x_prior[:, i] = kf.x_prior
    x_post[:, i] = kf.x
    
# Gather the results in a single np.array()
x_true = np.hstack(x_true) #hstack is semi-expensive, do this only once at the end
t = np.hstack(t)
dim_t = t.shape[0]

#%% Plot
ylabels = ["x1 [ft]", "x2 [ft/s]", "x3 []", "y []"]#
    
fig1, ax1 = plt.subplots(dim_x + 1, 1, sharex = True)
for i in range(dim_x): #plot true states and ukf's estimates
    ax1[i].plot(t, x_true[i, :], label = "True")
    ax1[i].plot(t_y, x_post[i, :], label = "UKF")
    ax1[i].plot(t_y, x_prior[i, :], label = "UKF-prior")
    ax1[i].set_ylabel(ylabels[i])
ax1[-1].set_xlabel("Time [s]")
#Plot measurements
ax1[-1].plot(t_y, y[0,:], marker = "x", markersize = 3, linewidth = 0, label = "y")
ax1[-1].set_ylabel(ylabels[-1])
ax1[0].legend()