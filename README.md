# Systematic Estimation of Noise Statistics for Nonlinear Kalman Filters
 
 Code for the paper "Systematic Estimation of Noise Statistics for Nonlinear Kalman Filters." The paper explores tuning strategies for nonlinear kalman filters based on parametric uncertainty. Parametric uncertainty is translated to noise statistics by either i) generalized unscented transformation, ii) latin hypercube sampling or iii) monte carlo simulations (a reference approach). Please refer to the paper for further information.
 
 What to run:

-"main_fb.py" contains code for running the simulation N times for the 5 filters (N=100 but you can change this). The run time depends on the number of sample point for the sampling based filters (N_lhs, N_mc and N_mcm), but it does not take many minutes.
 
 -"main_fb_different_N_LHS_w_mcm.py" contains the same code as above, but N_lhs and N_mc is adjusted to check how the distribution of the cost function decreases when the sampling number increases. This takes a long time (more than 12h) to run. The results are saved to the .pickle and .npy files in the "scripts" folder.
 
 -If you just want to plot the results from "main_fb_different_N_LHS_w_mcm.py" without running the simulation, run "plot_nlhs-many_iterations.py". It uses the .pickle and .npy files which are already in the directory.
 
 
