# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:37:15 2021

@author: halvorak
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

with open('mean_cost_5filters.pickle', 'rb') as handle:
    mean_cost = pickle.load(handle)   
with open('sigma_cost_5filters.pickle', 'rb') as handle:
    sigma_cost = pickle.load(handle)
N_LHS_list = np.load("N_lhs_and_mc_list_5filters.npy")

dim_x = 3

font = {'size'   : 14}

matplotlib.rc('font', **font)


if len(N_LHS_list) >= 2:
    labels_method = [r"GenUT", "LHS", "MC", "MCm", "Fixed"]
    fig_n_m, ax_n_m = plt.subplots(dim_x,1, sharex=True)
    ylabels_mu = [r"$\mu^{cost}_{x_1}$", r"$\mu^{cost}_{x_2}$", r"$\mu^{cost}_{x_3}$"]
    for i in range(dim_x):
        for j in range(len(labels_method)):
            ax_n_m[i].plot(N_LHS_list, 
                            mean_cost["x"+str(i+1)][j, :], 
                            label = labels_method[j], marker = "x")
        k = i+1
        ax_n_m[i].set_ylabel(ylabels_mu[i])
    ax_n_m[-1].set_xlabel(r"$N_{LHS}$ and $N_{MC}$")
    ax_n_m[0].legend(ncol = 3,
                      frameon = False)
    
    
    fig_n_s, ax_n_s = plt.subplots(dim_x,1, sharex = True)
    ylabels = [r"$\sigma^{cost}_{x_1}$", r"$\sigma^{cost}_{x_2}$", r"$\sigma^{cost}_{x_3}$"]
    for i in range(dim_x):
        for j in range(len(labels_method)):
            if j >= 3:
                continue
            ax_n_s[i].plot(N_LHS_list, 
                           sigma_cost["x"+str(i+1)][j, :], 
                           label = labels_method[j], marker = "x")
        
        ax_n_s[i].set_ylabel(ylabels[i])
    ax_n_s[-1].set_xlabel(r"$N_{LHS}$ and $N_{MC}$")
    ax_n_s[0].legend(ncol = 3,
                     frameon = False)