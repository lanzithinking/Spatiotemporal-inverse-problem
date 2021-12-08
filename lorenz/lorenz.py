#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The inverse problem of Lorenz system of differential equations
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Shuyi Li"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

-------------------------------------------------------------------------
--To run demo:                     python Lorenz.py # to compare with the finite difference method
--To define a class object:  e.g.  lrz=Lorenz(args)
--To obtain geometric quantities:  loglik,agrad,HessApply,eigs = lrz.get_geom(args) # misfit value, gradient, metric action and eigenpairs of metric resp.
                                   which calls _get_misfit, _get_grad, and _get_HessApply resp.
---------------------------------------------------------------
https://github.com/lanzithinking/Spatiotemporal-inverse-problem
'''
# import modules
import numpy as np
from scipy import integrate
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys, os

from ode import *
from prior import *
from misfit import *

STATE=0; PARAMETER=1; ADJOINT=2

class Lorenz:
    def __init__(self, num_traj=1, ode_params=None, prior_params=None, obs_times=None, **kwargs):
        """
        Initialize the Lorenz63 inverse problem consisting of the ODE model, the prior model and the misfit (likelihood) model.
        """
        self.num_traj = num_traj
        self.ode_params = {'sigma':10.0, 'beta':8./3, 'rho':28.0} if ode_params is None else ode_params
        self.x0 = kwargs.pop('ode_init' ,-15 + 30 * np.random.random((self.num_traj, 3)))
        self.prior_params = {'mean':[1.8, 1.2, 3.3], 'std':[1.0, 0.5, 0.15]} if prior_params is None else prior_params
        if obs_times is None:
            t_init = kwargs.pop('t_init',0.)
            t_final = kwargs.pop('t_final',4.)
            time_res = kwargs.pop('time_res',100)
            self.obs_times = np.linspace(t_init, t_final, time_res)
        else:
            self.obs_times = obs_times
        
        # define the inverse problem with ode, prior, and misfit
        seed = kwargs.pop('seed',2021)
        self.setup(seed,**kwargs)
        # initialize the joint state vector
        self.x = [[],]*3
    
    def setup(self,seed=2021,**kwargs):
        """
        Set up ode, prior, likelihood (misfit: -log(likelihood)) and posterior
        """
        # set (common) random seed
        np.random.seed(seed)
        sep = "\n"+"#"*80+"\n"
        # set ode
        print(sep, 'Define the Lorenz63 ODE model.', sep)
        self.ode = lrz63(self.x0, **self.ode_params, **kwargs)
        # set prior
        print(sep, 'Specify the prior model.', sep)
        self.prior = prior(**self.prior_params)
        # set misfit
        print(sep, 'Obtain the likelihood model.', sep)
        self.misfit = misfit(ode=self.ode, obs_times=self.obs_times, **kwargs)
#         # set low-rank approximate Gaussian posterior
#         print(sep, 'Set the approximate posterior model.', sep)
#         self.post_Ga = Gaussian_apx_posterior(self.prior, eigs='hold')
    
    def _get_misfit(self, parameter=None, MF_only=True):
        """
        Compute the misfit for given parameter.
        """
        if parameter is None:
            parameter=self.prior.mean
        self.x[PARAMETER] = np.exp(parameter)
        self.x[STATE] = self.ode.solveFwd(params=self.x[PARAMETER], t=self.misfit.obs_times)
        msft = self.misfit.cost(self.x[STATE][0])
        if not MF_only: msft += self.prior.cost(parameter)
        return msft
    
    def _get_grad(self, parameter=None, MF_only=True):
        """
        Compute the gradient of misfit (default), or the gradient of negative log-posterior for given parameter.
        """
        if parameter is None:
            parameter=self.prior.mean
        self.x[PARAMETER] = np.exp(parameter)
        x = self.x[STATE][0]
        self.x[ADJOINT] = self.ode.solveAdj(self.x[PARAMETER], self.misfit.obs_times, x, self.misfit, cont_soln=self.x[STATE][1])
        grad = np.sum(self.x[ADJOINT]*np.stack([-(x[:,:,1]-x[:,:,0]), x[:,:,2], -x[:,:,0]], axis=-1), axis=(0,1))
        if not MF_only: grad += self.prior.grad(parameter)
        return grad

    def _get_HessApply(self, parameter=None, MF_only=True):
        """
        Compute the Hessian apply (action) for given parameter,
        default to the Gauss-Newton approximation.
        """
        raise NotImplementedError('HessApply not implemented.')
    
    def get_geom(self,parameter=None,geom_ord=[0],**kwargs):
        """
        Get necessary geometric quantities including log-likelihood (0), adjusted gradient (1), 
        Hessian apply (1.5), and its eigen-decomposition using randomized algorithm (2).
        """
        if parameter is None:
            parameter=self.prior.mean
        loglik=None; agrad=None; HessApply=None; eigs=None;
        
        # get log-likelihood
        if any(s>=0 for s in geom_ord):
            loglik = -self._get_misfit(parameter, **kwargs)
        
        # get gradient
        if any(s>=1 for s in geom_ord):
            agrad = -self._get_grad(parameter, **kwargs)
        
        # get Hessian Apply
        if any(s>=1.5 for s in geom_ord):
            pass
        
        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        if any(s>1 for s in geom_ord):
            pass
        
        return loglik,agrad,HessApply,eigs
    
    def get_eigs(self,parameter=None,whitened=False,**kwargs):
        """
        Get the eigen-decomposition of Hessian action directly using randomized algorithm.
        """
        raise NotImplementedError('eigs not implemented.')
    
    def get_MAP(self,init='current',preconditioner='posterior',SAVE=False):
        """
        Get the maximum a posterior (MAP).
        """
        raise NotImplementedError('MAP not implemented.')
    
    def _check_folder(self,fld_name='result'):
        """
        Check the existence of folder for storing result and create one if not
        """
        if not hasattr(self, 'savepath'):
            cwd=os.getcwd()
            self.savepath=os.path.join(cwd,fld_name)
        if not os.path.exists(self.savepath):
            print('Save path does not exist; created one.')
            os.makedirs(self.savepath)
    
    def test(self,h=1e-4):
        """
        Demo to check results with the adjoint method against the finite difference method.
        """
        # random sample parameter
        parameter = self.prior.sample(add_mean=False)
        
        # MF_only = True
        import time
        # obtain the geometric quantities
        print('\n\nObtaining geometric quantities with Adjoint method...')
        start = time.time()
        loglik,grad,_,_ = self.get_geom(parameter,geom_ord=[0,1])
        end = time.time()
        print('Time used is %.4f' % (end-start))
        
        # check with finite difference
        print('\n\nTesting against Finite Difference method...')
        start = time.time()
        # random direction
        v = self.prior.sample(add_mean=False)
        ## gradient
        print('\nChecking gradient:')
        parameter_p = parameter + h*v
        loglik_p = -self._get_misfit(parameter_p)
#         parameter_m = parameter - h*v
#         loglik_m = -self._get_misfit(parameter_m)
        dloglikv_fd = (loglik_p-loglik)/h
        dloglikv = np.sum(grad*v)
        rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/np.linalg.norm(v)
        print('Relative difference of gradients in a random direction between adjoint and finite difference: %.10f' % rdiff_gradv)
        end = time.time()
        print('Time used is %.4f' % (end-start))
    
if __name__ == '__main__':
    # set up random seed
    seed=2021
    np.random.seed(seed)
    # define Bayesian inverse problem
    num_traj = 1
    t_init = 1000
    t_final = 1100
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = False
    lrz = Lorenz(num_traj=num_traj, obs_times=obs_times, avg_traj=avg_traj, seed=seed, STlik=True)
    # test
    lrz.test(1e-4)
    # obtain MAP
    # map_v = lrz.get_MAP(init='zero',SAVE=True)
    # # compare it with the truth
    # true_param = [10.0, 8./3, 28.0]
    # relerr = np.linalg.norm(map_v-true_param)/np.linalg.norm(true_param)
    # print('Relative error of MAP compared with the truth %.2f%%' % (relerr*100))
    # # report the minimum cost
    # min_cost = lrz._get_misfit(map_v)
    # print('Minimum cost: %.4f' % min_cost)
