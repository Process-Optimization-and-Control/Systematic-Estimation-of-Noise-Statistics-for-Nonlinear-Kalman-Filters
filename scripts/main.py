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
from myFilter import sigma_points
# from myFilter import UKF_constrained

#Self-written modules
import sigma_points_classes as spc
import unscented_transformation as ut
import utils_inverted_pendulum as utils_ip

#%% Define dimensions and initialize arrays
# utils_ip.uncertainty_venturi()
#Get nominal parameters, initial values and assign space for variables
par, Q_nom, R_nom = utils_ip.get_nominal_values()
x0 = utils_ip.get_x0_inverted_pendulum()
dim_x = x0.shape[0]
dt_y = 5e-3 # [s] <=> 5 ms. Measurement frequency
dt = 1e-4 #[s] - discretization time for integration
t = np.linspace(0,2, int(2/dt), endpoint = True)
dim_t = t.shape[0]
dim_y = int(np.ceil((t[-1] - t[0]) / dt_y) + 1)
t_y = np.linspace(t[0], t[-1], dim_y)
x_true = np.zeros((dim_x, dim_t))
x_post = np.zeros((dim_x, dim_t))
u = np.zeros(dim_t)
u_kf = np.zeros(dim_t)
x_true[:, 0] = x0
u[0] = 40*x_true[2, 0] # x_true[2,:] is theta
#%% Simulate plant and measurements

points = sigma_points.JulierSigmaPoints(dim_x,
                                        kappa = 3-dim_x)
x_post[:, 0] = x0 #know initial state perfectly
kf = UKF.UnscentedKalmanFilter(dim_x = dim_x, 
                               dim_z = dim_y, 
                               dt = dt, 
                               hx = lambda x: x[0], 
                               fx = lambda x: utils_ip.ode_model_plant(x, t, u[0], par)* dt,
                               points = points)

#Generate measurement vectors
v = np.random.normal(loc = 0, scale = np.sqrt(R_nom), size = dim_y) #white noise
y = v #initialize y with measurement noise
y[0] += x_true[0, 0]
k = 1
for i in range(1, dim_t):
    dxdt = utils_ip.ode_model_plant(x_true[:, i-1], t, u[i-1], par)
    x_true[:, i] = x_true[:, i-1] + dxdt*dt
    u[i] = 40*x_true[2, i]
    if t[i] >= t_y[k]: #we got now a new measurement
        y[k] += x_true[0, i] # measures d. The noise is already added
        k += 1




#%% Plot
x_true[2:,:] = np.rad2deg(x_true[2:,:])
fig1, ax1 = plt.subplots(dim_x, 1, sharex = True)
# ylabels_rad = ["d [rad]", r"$\dot{d}$ [rad/s]", r"$\theta$ [rad]", r"$\dot{\theta}$ [rad/s]"]
ylabels_deg = ["d [m]", r"$\dot{d}$ [m/s]", r"$\theta$ [deg]", r"$\dot{\theta}$ [deg/s]"]
ylabels = ylabels_deg

for i in range(dim_x):
    ax1[i].plot(t, x_true[i, :], label = "True")
    ax1[i].set_ylabel(ylabels[i])
ax1[-1].set_xlabel("Time [s]")

#Plot measurements
ax1[0].plot(t_y, y, marker = "x", markersize = 3, linewidth = 0, label = "y")
ax1[0].legend()
    
