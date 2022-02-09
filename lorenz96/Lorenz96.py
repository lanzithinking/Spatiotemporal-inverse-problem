#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
The inverse problem of Lorenz96 system of differential equations
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Shuyi Li"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

-------------------------------------------------------------------------
--To run demo:                     python Rossler.py # to compare with the finite difference method
--To define a class object:  e.g.  rsl=Rossler(args)
--To obtain geometric quantities:  loglik,agrad,HessApply,eigs = rsl.get_geom(args) # misfit value, gradient, metric action and eigenpairs of metric resp.
                                   which calls _get_misfit, _get_grad, and _get_HessApply resp.
---------------------------------------------------------------
https://github.com/lanzithinking/Spatiotemporal-inverse-problem
'''
# import modules
import numpy as np
from scipy import integrate, optimize
import sys, os

from ode import *
from prior import *
from misfit import *

STATE=0; PARAMETER=1; ADJOINT=2
class Lorenz96:
    def __init__(self, num_traj=1, ode_params=None, prior_params=None, obs_times=None, **kwargs):
        """
        Initialize the Lorenz96 inverse problem consisting of the ODE model, the prior model and the misfit (likelihood) model.
        """
        self.num_traj = num_traj
        self.ode_params = {'h':1, 'F':10, 'logc':np.log(10),'b':10} if ode_params is None else ode_params
        self.L = kwargs.pop('L',10)
        self.K = kwargs.pop('K',36)
        self.n = (self.L+1)*self.K
        self.x0 = kwargs.pop('ode_init', np.random.RandomState(kwargs.pop('randinit_seed',2021)).random((self.num_traj, self.n))) # fixed initial condition
        self.prior_params = {'mean':[0, 10, 2, 5], 'std':[1, 10**.5, 0.1**.5, 10**.5]} if prior_params is None else prior_params
  
        if obs_times is None:
            t_init = kwargs.pop('t_init',0.)
            t_final = kwargs.pop('t_final',10.)
            time_res = kwargs.pop('time_res',200)
            self.obs_times = np.linspace(t_init, t_final, time_res)
        else:
            self.obs_times = obs_times
        
        # define the inverse problem with ode, prior, and misfit
        seed = kwargs.pop('seed',2021)
        self.setup(seed,**kwargs)
        # initialize the joint state vector
        # self.x = [[],]*3
        self.x = [self.x0[:,None,:], list(self.ode_params), []]
        
    def setup(self,seed=2021,**kwargs):
        """
        Set up ode, prior, likelihood (misfit: -log(likelihood)) and posterior
        """
        # set (common) random seed
        np.random.seed(seed)
        sep = "\n"+"#"*80+"\n"
        # set ode
        print(sep, 'Define the Lorenz96 ODE model.', sep)
        self.ode = lorenz96(self.x0, t=self.obs_times, L=self.L, K=self.K, **self.ode_params, **kwargs)
        # set prior
        print(sep, 'Specify the prior model.', sep)
        self.prior = prior(**self.prior_params)
        # set misfit
        print(sep, 'Obtain the likelihood model.', sep)
        self.misfit = misfit(ode=self.ode, obs_times=self.obs_times, **kwargs)
#         # set low-rank approximate Gaussian posterior
#         print(sep, 'Set the approximate posterior model.', sep)
#         self.post_Ga = Gaussian_apx_posterior(self.prior, eigs='hold')
    
    def _get_misfit(self, parameter=None, MF_only=True, warm_start=False):
        """
        Compute the misfit for given parameter.
        """
        if parameter is None:
            parameter=self.prior.mean
        self.x[PARAMETER] = (parameter)
        if warm_start: self.ode.x0 = self.x[STATE][:,-1,:]
        self.x[STATE] = self.ode.solveFwd(params=self.x[PARAMETER], t=self.misfit.obs_times)
        msft = self.misfit.cost(self.x[STATE])
        if not MF_only: msft += self.prior.cost(parameter)
        return msft
        
if __name__ == '__main__':
    # set up random seed
    seed=2021
    np.random.seed(seed)  
    
    num_traj = 1
    t_init = 0
    t_final = 10
    time_res = 200
    obs_times = np.linspace(t_init, t_final, time_res)
    L, K = 2, 2
    n = (L+1) * K
    x0 = np.random.randn(num_traj, n)
    avg_traj = True
    var_out = True#'cov'    
    STlik = False
    
    rsl96 = Lorenz96(ode_init=x0, obs_times=obs_times, L=L, K=K, avg_traj=avg_traj, var_out=var_out, seed=2021, STlik=STlik)
    xts, yts = rsl96.ode.solve()
    
    
    
    
