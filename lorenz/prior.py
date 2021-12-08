#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The prior of Lorenz inverse problem 
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Shuyi Li"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

'''
# import modules
import numpy as np
import matplotlib.pyplot as plt

class prior:
    """
    (Log-Normal) prior for parameters in Lorenz63 inverse problem.
    """
    def __init__(self, mean=[1.8, 1.2, 3.3], std=[1.0, 0.5, 0.15]):
        """
        mean and standard deviation for logarithms of (sigma, beta, rho)
        """
        self.mean = mean
        self.std = std
        self.d = self.mean.shape[1] if np.ndim(self.mean)>1 else len(self.mean)
    
    def cost(self, x):
        """
        negative log-prior
        """
        dx = x - self.mean
        reg = .5* np.sum((dx/self.std)**2)
        return reg
    
    def grad(self, x):
        """
        gradient of negative log-prior
        """
        dx = x - self.mean
        g = dx/self.std
        return g
    
    def sample(self, n=1, add_mean=True):
        """
        Generate a prior sample
        """
        z = np.random.randn(n,self.d)
        u = z * self.std
        if add_mean:
            u += self.mean
        return u.squeeze()
    
    def logpdf(self, u, add_mean=True, grad=False):
        """
        Compute the logarithm of prior density (and its gradient) of parameter u.
        """
        if add_mean:
            u -= self.mean
        
        gradpri = -u/self.std
        logpri = -.5* np.sum(gradpri**2)
        if grad:
            return logpri, gradpri
        else:
            return logpri
        
    def C_act(self, u_actedon, comp=1):
        """
        Compute operation of C^comp on vector u: u --> C^comp * u
        """
        if comp==0:
            return u_actedon
        else:
            Cu = np.pow(self.std,comp*2)*u_actedon
            return Cu

if __name__ == '__main__':
    np.random.seed(2021)
    # define prior
    prior = prior()
    # tests
    u=prior.sample()
    logpri,gradpri=prior.logpdf(u, grad=True)
    print('The logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri,np.linalg.norm(gradpri)))
    