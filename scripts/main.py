# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:44:04 2021

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

# Did some modification to these packages
from myFilter import UKF
from myFilter import sigma_points as ukf_sp
# from myFilter import UKF_constrained

#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import utils_inverted_pendulum as utils_ip

#%% Define dimensions and initialize arrays
# utils_ip.uncertainty_venturi()
#Get nominal parameters, initial values and assign space for variables
par, Q_nom, R_nom = utils_ip.get_literature_values()

#%% Get other parameter values
par_dist, par_det, fig_p, ax_p = utils_ip.get_param_ukf_case1(
    std_dev_prct = 0.05, 
    plot_dist = True)
sigmas, w = utils_ip.get_sigmapoints_and_weights(par_dist)

mode_dist = np.zeros(len(par_dist))
par_true = par_det.copy()
par_kf = par_det.copy()
del par_det
list_dist_keys = list(par_dist.keys()) # list of keys. Useful later in the code
for key, dist in par_dist.items():
    #Need to find the mode numericlly
    mode = scipy.optimize.minimize(lambda theta: -dist.pdf(theta),
                                   dist.mean()) #use mean as theta0
    par_true[key] = mode.x
    par_kf[key] = dist.mean()

    print(f"{key}_true/{key}_KF: {par_true[key]/par_kf[key]}")

#%%

x0 = utils_ip.get_x0_inverted_pendulum()
dim_x = x0.shape[0]
dt_y = 5e-3 # [s] <=> 5 ms. Measurement frequency
dt = 1e-4 #[s] - discretization time for integration

t_end = 20
t_y = np.linspace(0, t_end, int(t_end/dt_y))
# t = np.linspace(t_y[0],t_y[0+1], int(dt_y/dt), endpoint = True)

dim_ty = t_y.shape[0]
dim_y = 1#t_y.shape[0]

x_true = [[] for _ in range(dim_ty-1)] #make a list of list
t = [[] for _ in range(dim_ty-1)] #make a list of list
x_post = np.zeros((dim_x, dim_ty))
x_post[:, 0] = x0
t_span = (t_y[0],t_y[1])
#%% Check we get yi for sigma points

fx_gen_Q = lambda si: utils_ip.fx_for_UT_gen_Q(si, list_dist_keys, t_span, x0, par_kf)
mean_ut, Q_ut = ut.unscented_transformation(sigmas, w, fx = fx_gen_Q)

# for i in range(len(w)):
#     si = sigmas[:, i]
#     yi = utils_ip.fx_for_UT_gen_Q(si, list_dist_keys, t_span, x0, par_kf)

#%% Simulate plant and measurements

t_span = (t_y[0],t_y[1])
points = ukf_sp.JulierSigmaPoints(dim_x,
                                  kappa = 0)#3-dim_x)
fx_ukf = lambda x, dt_kf: utils_ip.fx_ukf_ode(utils_ip.ode_model_plant, 
                                             t_span, 
                                             x,
                                             args_ode = (par_kf,)
                                             )
kf = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                                dim_z = dim_y, 
                                dt = 100, 
                                hx = lambda x: np.array([x[0]]), 
                                fx = fx_ukf,
                                points = points)
kf.x = x_post[:, 0]
kf.P = np.diag([1e-4,#1e-4 corresponds to std of 0,01m <=> 1 cm
                1e-8, #no uncertainty in velocty
                1e-3, #In radians.  1e-3 corresponds to 1,8deg in standard deviations, np.rad2deg(np.sqrt(1e-3))
                1e-8])
kf.Q = Q_nom
# kf.Q = Q_ut
kf.R = R_nom
#Generate measurement vectors
v = np.random.normal(loc = 0, scale = np.sqrt(R_nom), size = dim_ty) #white noise
y = v #initialize y with measurement noise
y[0] += x0[0]
for i in range(1,dim_ty):
    t_span = (t_y[i-1],t_y[i])
    res = scipy.integrate.solve_ivp(utils_ip.ode_model_plant, 
                                    t_span,#(t_y[i-1],t_y[i]), 
                                    x0, 
                                    # rtol = 1e-10,
                                    # atol = 1e-13
                                    args = (par_true,)
                                    )
    t[i-1] = res.t #add integration time to the list
    x_true[i-1] = res.y #add the interval to the full list
    
    #Prediction step of kalman filter
    fx_ukf = lambda x, dt_kf: utils_ip.fx_ukf_ode(utils_ip.ode_model_plant, 
                                             t_span, 
                                             x,
                                             args_ode = (par_kf,
                                                         None #control law for u. If None, it follows u = 40*x_kf[0]
                                                         # lambda theta: 0.,#sin_fx
                                                         # lambda theta: 1. #cos_fx
                                                         )
                                             )
    fx_gen_Q = lambda si: utils_ip.fx_for_UT_gen_Q(si, list_dist_keys, t_span, x_post[:, i-1], par_kf)
    mean_ut, Q_ut = ut.unscented_transformation(sigmas, w, fx = fx_gen_Q)
    # kf.Q = Q_ut
    
    kf.predict(fx = fx_ukf)
    
    #Get measurement
    x0 = res.y[:, -1] #starting point for next integration interval
    y[i] += x0[0] #add the measurement

    #Correction step of UKF
    kf.update(np.array([y[i]]))
    x_post[:, i] = kf.x
    
# Gather the results in a single np.array()
x_true = np.hstack(x_true) #hstack is semi-expensive, do this only once at the end
t = np.hstack(t)
dim_t = t.shape[0]

#%% Plot
use_rad = True
if use_rad:
    ylabels = ["d [m]", r"$\dot{d}$ [m/s]", r"$\theta$ [rad]", r"$\dot{\theta}$ [rad/s]"]
else: #use degrees
    x_true[2:,:] = np.rad2deg(x_true[2:,:])
    x_post[2:, :] = np.rad2deg(x_post[2:, :])
    ylabels= ["d [m]", r"$\dot{d}$ [m/s]", r"$\theta$ [deg]", r"$\dot{\theta}$ [deg/s]"]
    
fig1, ax1 = plt.subplots(dim_x, 1, sharex = True)
for i in range(dim_x): #plot true states and ukf's estimates
    ax1[i].plot(t, x_true[i, :], label = "True")
    ax1[i].plot(t_y, x_post[i, :], label = "UKF")
    ax1[i].set_ylabel(ylabels[i])
ax1[-1].set_xlabel("Time [s]")
#Plot measurements
ax1[0].plot(t_y, y, marker = "x", markersize = 3, linewidth = 0, label = "y")

ax1[0].legend()
    
