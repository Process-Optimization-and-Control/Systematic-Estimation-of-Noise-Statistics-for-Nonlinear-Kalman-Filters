# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:54:11 2021

@author: halvorak
"""

import numpy as np
import scipy.stats
# import matplotlib.pyplot as plt
# import colorcet as cc
# import pathlib
# import os
import scipy.linalg
# import matplotlib.patches as plt_patches
# import matplotlib.transforms as plt_transforms


class SigmaPoints():
    """
    Parent class when sigma points algorithms are constructed. All points tru to estimate mean and covariance of Y, where Y=f(X)
    """
    def __init__(self, n, sqrt_method=None):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is None. Method to calculate the square root of a matrix. If None is supplied, scipy.linalg.cholesky is used

        Returns
        -------
        None.

        """
        self.n = n
        if sqrt_method is None:
            self.sqrt = scipy.linalg.cholesky 
        else:
            self.sqrt = sqrt_method
        
    def num_sigma_points(self):
        """
        Returns the number of sigma points. Most algorithms return (2n+1) points, can be overwritten by child class

        Returns
        -------
        TYPE int
            DESCRIPTION. dim_sigma, number of sigma points

        """
        return 2*self.n + 1
    
    def is_matrix_pos_def(self, a_matrix):
        """
        Checks if a matrix is positive definite

        Parameters
        ----------
        a_matrix : TYPE np.array((n,n))
            DESCRIPTION. A matrix

        Returns
        -------
        TYPE bool
            DESCRIPTION. True if the matrix is pos def, else False

        """
        return np.all(np.linalg.eigvals(a_matrix) > 0)
    

    
class JulierSigmaPoints(SigmaPoints):
    """
    Implement the sigma points as described by Julier's original paper. It assumes that the distribtions are symmetrical.
    
    @TECHREPORT{Julier96ageneral,
    author = {Simon Julier and Jeffrey K. Uhlmann},
    title = {A General Method for Approximating Nonlinear Transformations of Probability Distributions},
    institution = {},
    year = {1996}
}
    
    """
    def __init__(self, n, kappa = 0., sqrt_method=None):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is None. Method to calculate the square root of a matrix. If None is supplied, scipy.linalg.cholesky is used
        kappa : TYPE, optional float
            DESCRIPTION. The default is 0. If set to (n-3), you minimize error in higher order terms.


        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        
        if not (kappa == (n-3)):
            print(f"warning: kappa is not set to kappa = (n-3) = {n-3}, which minimizes the fourth order mismatch. Proceeding with a value of kappa = {kappa}")
        self.kappa = kappa
        self.dim_sigma = self.num_sigma_points()
        
    # def compute_weights(self)
    def compute_sigma_points(self, mu, P):
        """
        Computes the sigma points based on Julier's paper

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt((n+kappa)P). Can be inspected if something goes wrong.

        """
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
            raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        
        n = self.n
        dim_sigma = self.dim_sigma
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        try:
            P_sqrt = self.sqrt((n+self.kappa)*P)
        except np.linalg.LinAlgError as LinAlgError:
            print(f"(n+kappa)P is not positive definite. Current value is (n+kappa)P = {(n+self.kappa)*P}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu + P_sqrt[i, :]
            sigmas[:, 1 + n + i] = mu - P_sqrt[i, :]
        
        return sigmas, P_sqrt
        
    def compute_weights(self):
        """
        Computes the weights

        Returns
        -------
        weights : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points

        """
        n = self.n
        dim_sigma = self.dim_sigma
        
        weights = np.array([1/(2*(n + self.kappa)) for i in range(dim_sigma)])
        weights[0] = self.kappa/(n + self.kappa)
        return weights
        

class GenUTSigmaPoints(SigmaPoints):
    """
    Implement the sigma points as described by Ebeigbe. Distributions does NOT need to be symmetrical.
    
    @article{EbeigbeDonald2021AGUT,
abstract = {The unscented transform uses a weighted set of samples called sigma points to propagate the means and covariances of nonlinear transformations of random variables. However, unscented transforms developed using either the Gaussian assumption or a minimum set of sigma points typically fall short when the random variable is not Gaussian distributed and the nonlinearities are substantial. In this paper, we develop the generalized unscented transform (GenUT), which uses adaptable sigma points that can be positively constrained, and accurately approximates the mean, covariance, and skewness of an independent random vector of most probability distributions, while being able to partially approximate the kurtosis. For correlated random vectors, the GenUT can accurately approximate the mean and covariance. In addition to its superior accuracy in propagating means and covariances, the GenUT uses the same order of calculations as most unscented transforms that guarantee third-order accuracy, which makes it applicable to a wide variety of applications, including the assimilation of observations in the modeling of the coronavirus (SARS-CoV-2) causing COVID-19.},
journal = {ArXiv},
year = {2021},
title = {A Generalized Unscented Transformation for Probability Distributions},
language = {eng},
address = {United States},
author = {Ebeigbe, Donald and Berry, Tyrus and Norton, Michael M and Whalen, Andrew J and Simon, Dan and Sauer, Timothy and Schiff, Steven J},
issn = {2331-8422},
}


    """
    def __init__(self, n, sqrt_method=None):
        """
        Init

        Parameters
        ----------
        n : TYPE int
            DESCRIPTION. Dimension of x
        sqrt_method : TYPE, optional function
            DESCRIPTION. The default is None. Method to calculate the square root of a matrix. If None is supplied, scipy.linalg.cholesky is used

        Returns
        -------
        None.

        """
        super().__init__(n, sqrt_method = sqrt_method)
        self.dim_sigma = self.num_sigma_points()
        
    def compute_scaling_and_weights(self, P, S, K, s1 = None):
        """
        Computes the scaling parameters s and the weights w

        Parameters
        ----------
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance of X
        S : TYPE np.array(n,)
            DESCRIPTION. 3rd central moment of X. Can be computed by scipy.stats.moments(data, moment=3)
        K : TYPE np.array(n,)
            DESCRIPTION. 4th central moment of X. Can be computed by scipy.stats.moments(data, moment=4)
        s1 : TYPE, optional np.array(n,)
            DESCRIPTION. The default is None. First part of scaling arrays. s1> 0 for every element. If None, algorithm computes the suggested values in the article.

        Raises
        ------
        ValueError
            DESCRIPTION. Dimension mismatch

        Returns
        -------
        s : TYPE np.array(2n,)
            DESCRIPTION. Scaling values
        w : TYPE np.array(dim_sigma,)
            DESCRIPTION. Weights for every sigma points.

        """
        
        sigma = np.sqrt(np.diag(P)) #standard deviation of each state
        S_std = np.divide(S, np.power(sigma, 3))
        K_std = np.divide(K, np.power(sigma, 4))
        
        if s1 is None: #create s (s.shape = (n,))
            s1 = self.select_s1_to_match_kurtosis(S_std, K_std)
        
        if (s1.shape[0] != S.shape[0]):
            raise ValueError("Dimension of s is wrong")
        
        #create the next values for s, total dim is 2n+1
        s2 = s1 + S_std
        w2 = np.divide(1, np.multiply(s2, (s1 + s2)))
        w1 = np.multiply(np.divide(s2, s1), w2)
        w = np.concatenate((np.array([0]), w1, w2))
        w[0] = 1 - np.sum(w[1:])
        s = np.concatenate((s1, s2))
        return s, w
    
    def select_s1_to_match_kurtosis(self, S_std, K_std):
        """
        Computes the first part of the scaling array by the method suggested in the paper.

        Parameters
        ----------
        S_std : TYPE np.array(n,)
            DESCRIPTION. Scaled 3rd central moment
        K_std : TYPE np.array(n,)
            DESCRIPTION. Scaled 4th central moment

        Returns
        -------
        s1 : TYPE np.array(n,)
            DESCRIPTION. First part of the scaling value array

        """
        s1 = .5*(-S_std + np.sqrt(4*K_std - 3*np.square(S_std)))
        return s1
    
    def compute_sigma_points(self, mu, P, s):
        """
        Computes the sigma points

        Parameters
        ----------
        mu : TYPE np.array(n,)
            DESCRIPTION. Mean value of X 
        P : TYPE np.array(n,n)
            DESCRIPTION. Covariance matrix of X

        Raises
        ------
        ValueError
            DESCRIPTION. Shapes are wrong
        LinAlgError
            DESCRIPTION. P is not positiv definite and symmetric

        Returns
        -------
        sigmas : TYPE np.array(n, dim_sigma)
            DESCRIPTION. sigma points
        P_sqrt : TYPE np.array(n,n)
            DESCRIPTION. sqrt(P). Can be inspected if something goes wrong.

        """
        if not self.n == mu.shape[0]:
            raise ValueError(f" self.n = {self.n} while mu.shape = {mu.shape}. mu.shape[0] must match self.n!")
        
        if not ((self.n == P.shape[0]) and (self.n == P.shape[1])):
            raise ValueError(f"P.shape = {P.shape}, it must be ({self.n, self.n})")
        
        n = self.n
        dim_sigma = self.dim_sigma
        
        sigmas = np.zeros((n, dim_sigma))
        sigmas[:, 0] = mu
        
        try:
            P_sqrt = self.sqrt(P)
        except np.linalg.LinAlgError as LinAlgError:
            print(f"P is not positive definite. Current value is P = {P}")
            raise LinAlgError
        
        for i in range(n):
            sigmas[:, 1 + i] = mu - s[i]*P_sqrt[i, :]
            sigmas[:, 1 + n + i] = mu + s[n + i]*P_sqrt[i, :]
        
        return sigmas, P_sqrt

# def unscented_transform(sigmas, w, fx = None):
#     """
#     Calculates mean and covariance of a nonlinear function by the unscented transform. Every sigma point is propagated through the function, and combined with their weights the mean and covariance is calculated.

#     Parameters
#     ----------
#     sigmas : TYPE np.ndarray(n, dim_sigma)
#         DESCRIPTION. Array of sigma points. Each column contains a sigma point
#     w : TYPE np.array(dim_sigma,)
#         DESCRIPTION. Weights of each sigma point.
#     fx : TYPE, optional function
#         DESCRIPTION. The default is None. The non-linear function which the RV is propagated through. If None is supplied, the identity function is used.

#     Returns
#     -------
#     mean : TYPE np.array(n,)
#         DESCRIPTION. Mean value of Y=f(X), X is a random variable (RV) 
#     Py : TYPE np.array(n,n)
#         DESCRIPTION. Covariance matrix, cov(Y) where Y=f(X)

#     """
#     if fx is None:
#         fx = lambda x: x
    
#     (n, dim_sigma) = sigmas.shape
    
#     yi = np.zeros(sigmas.shape)
#     for i in range(dim_sigma):
#         yi[:, i] = fx(sigmas[:, i])
#     mean = np.dot(yi, w)
    
#     Py = np.zeros((n,n))
#     for i in range(dim_sigma):
#         Py += w[i]*np.dot((yi[:, i] - mean).reshape(-1,1), 
#                           (yi[:, i] - mean).reshape(-1,1).T)
    
#     return mean, Py



