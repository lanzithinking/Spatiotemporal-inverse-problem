#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The prior of Chen inverse problem 
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Shuyi Li"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

'''
# import modules
import numpy as np
import matplotlib.pyplot as plt

class prior:
    """
    (Log-Normal) prior for parameters in Chen inverse problem.
    """
    def __init__(self, mean=[4.0, 1.2, 3.3], std=[0.4, 0.5, 0.15]):
        """
        mean and standard deviation for logarithms of (a, b, c)
        """
        self.mean = mean if type(mean) is np.ndarray else np.array(mean)
        self.std = std if type(std) is np.ndarray else np.array(std)
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
        g = dx/self.std**2
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
        
        gradpri = -u/self.std**2
        logpri = -.5* np.sum(-u*gradpri)
        if grad:
            return logpri, gradpri
        else:
            return logpri
        
    def C_act(self, u_actedon=None, comp=1):
        """
        Compute operation of C^comp on vector u: u --> C^comp * u
        """
        if u_actedon is None:
            return np.diag(np.power(self.std,comp*2))
        else:
            if comp==0:
                return u_actedon
            else:
                Cu = np.expand_dims(np.power(self.std,comp*2),axis=list(range(1,u_actedon.ndim)))*u_actedon
                return Cu

if __name__ == '__main__':
    np.random.seed(2021)
    # define prior
    prior = prior()
    # tests
    u=prior.sample()
    logpri,gradpri=prior.logpdf(u, grad=True)
    print('The logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri,np.linalg.norm(gradpri)))
    