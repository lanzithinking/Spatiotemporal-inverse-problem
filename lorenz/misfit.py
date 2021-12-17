#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The misfit of Lorenz inverse problem 
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
from scipy import integrate
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import os, pickle

from ode import *

import sys, os, pickle
sys.path.append( "../" )
from util.stgp.GP import GP
from util.stgp.STGP import STGP
from util.stgp.STGP_mg import STGP_mg

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class misfit:
    """
    Misfit for Lorenz63 inverse problem.
    """
    def __init__(self, ode=None, obs_times=None, **kwargs):
        """
        ode: Lorenz63 object
        obs_times: time points where observations are taken
        true_params: true parameters of Lorenz63 for which observations are obtained
        avg_traj: whether to average the solution trajectories - False (whole trajectories); True (average trajectories); 'aug' (augment averaged trajectories with 2nd order interactions)
        STlik: whether to use spatiotemporal models; has to be used with 'avg_traj=False'
        stgp: spatiotemporal GP kernel
        """
        self.ode = lrz63(**kwargs) if ode is None else ode
        if obs_times is None:
            t_init = kwargs.pop('t_init',0.)
            t_final = kwargs.pop('t_final',4.)
            time_res = kwargs.pop('time_res',100)
            self.obs_times = np.linspace(t_init, t_final, time_res)
        else:
            self.obs_times = obs_times
        self.true_params = kwargs.pop('true_params',{'sigma':10.0, 'beta':8./3, 'rho':28.0})
        self.avg_traj = kwargs.pop('avg_traj',True) # False, True, or 'aug'
        # get observations
        self.obs, self.nzvar = self.get_obs(**kwargs)
        
        self.STlik = kwargs.pop('STlik',False)
        if self.STlik:
            if self.avg_traj:
                warnings.warn('Spatiotemporal models not available for time-averaged trajectories!')
                self.avg_traj = False
            self.stgp = kwargs.get('stgp')
            if self.stgp is None:
                # define STGP kernel for the likelihood (misfit)
                # -- model 0 -- separable STGP
                # self.stgp=STGP(spat=self.obs.mean(axis=0).T, temp=self.obs_times, opt=kwargs.pop('ker_opt',0), jit=1e-2)
                C_x=GP(self.obs.mean(axis=0).T, l=.5, sigma2=np.diag(np.sqrt(self.nzvar.mean(axis=0))), store_eig=True, jit=1e-2)
                C_t=GP(self.obs_times, store_eig=True, l=.2, sigma2=np.sqrt(self.nzvar.mean(axis=0).sum()))#, ker_opt='matern',nu=.5)
                # C_x=GP(self.obs.mean(axis=0).T, l=.4, jit=1e-2, sigma2=.1, store_eig=True)
                # C_t=GP(self.obs_times, jit=1e-2, store_eig=True, l=.1, sigma2=.1, ker_opt='matern',nu=.5)
                self.stgp=STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',0), spdapx=False)
                # C_x=GP(self.obs.mean(axis=0).T, l=.5, sigma2=.1, store_eig=True)
                # C_t=GP(self.obs_times, store_eig=True, l=.2, sigma2=1.)#, ker_opt='matern',nu=.5)
                # self.stgp=STGP_mg(STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',2), spdapx=False), K=1, nz_var=self.nzvar.mean(axis=0).sum(), store_eig=True)
                # -- model 1 -- full STGP
                # self.stgp=GP(self.obs.mean(axis=0).flatten(), l=.5, store_eig=True, jit=1e-3)
    
    def get_obs(self, **kwargs):
        """
        Obtain observations
        """
        fld=kwargs.pop('obs_file_loc',os.getcwd())
        try:
            f=open(os.path.join(fld,'Lorenz_obs_'+{True:'avg',False:'full','aug':'avgaug'}[self.avg_traj]+'_traj.pckl'),'rb')
            obs, nzvar=pickle.load(f)
            f.close()
            print('Observation file '+'Lorenz_obs_'+{True:'avg',False:'full','aug':'avgaug'}[self.avg_traj]+'_traj.pckl'+' has been read!')
        except Exception as e:
            print(e)
            ode=kwargs.pop('ode',self.ode)
            params=kwargs.pop('params',tuple(self.true_params.values()))
            t=self.obs_times
            sol=ode.solve(params=params, t=t)
            obs, nzvar=self.observe(sol, var_out=True)
            if kwargs.pop('save_obs',True):
                f=open(os.path.join(fld,'Lorenz_obs_'+{True:'avg',False:'full','aug':'avgaug'}[self.avg_traj]+'_traj.pckl'),'wb')
                pickle.dump([obs, nzvar], f)
                f.close()
        return obs, nzvar
    
    def observe(self, sol=None, var_out=False):
        """
        Observation operator
        """
        if sol is None:
            sol = self.ode.solve(params=tuple(self.true_params.values()), t=self.obs_times)
        assert np.ndim(sol)==3, 'The shape of solution should be (num_traj, time_res, 3)!'
        if var_out: nzvar = sol.var(axis=1)
        if self.avg_traj:
            obs = sol.mean(axis=1) # (num_traj, 3)
            ext = []; vxt = []
            if self.avg_traj=='aug':
                for i in range(sol.shape[0]):
                    ext.append(pdist(sol[i].T, metric=lambda u,v:np.mean(u*v)))
                    if var_out: vxt.append(pdist(sol[i].T, metric=lambda u,v:np.var(u*v)))
                obs = np.hstack([obs,np.mean(sol**2,axis=1),np.array(ext)]) # (num_traj, 9)
                if var_out: nzvar = np.hstack([nzvar,np.var(sol**2,axis=1),np.array(vxt)])
        else:
            obs = sol # (num_traj, time_res, 3)
        return (obs, nzvar) if var_out else obs
    
    def cost(self, sol=None, option='nll'):
        """
        Compute misfit
        option: return negative loglike ('nll') or (postive) quadratic form (quad) where loglike = halfdelt+quad
        """
        if sol is None:
            sol = self.ode.solve(params=tuple(self.true_params.values()), t=self.obs_times)
        dif_obs = self.observe(sol=sol) - self.obs
        if self.STlik:
            logpdf,half_ldet = self.stgp.matn0pdf(np.swapaxes(dif_obs,0,2).reshape((np.prod(dif_obs.shape[1:]),-1),order='F'))
            res = {'nll':-logpdf, 'quad':-(logpdf - half_ldet), 'both':[-logpdf,-(logpdf - half_ldet)]}[option]
        else:
            dif2_obs = dif_obs**2
            if np.ndim(dif_obs)==3: dif2_obs = dif2_obs.sum(axis=1)
            res = np.sum(dif2_obs/self.nzvar)/2
        return res
        
    def grad(self, sol=None, wrt='obs_sol'):
        """
        Compute the gradient of misfit
        """
        if sol is None:
            sol = self.ode.solve(params=tuple(self.true_params.values()), t=self.obs_times)
        dif_obs = self.observe(sol=sol) - self.obs
        if self.STlik:
            g = self.stgp.solve(np.swapaxes(dif_obs,0,2).reshape((np.prod(dif_obs.shape[1:]),-1),order='F')).T.reshape((dif_obs.shape[0],dif_obs.shape[2],dif_obs.shape[1]),order='F').swapaxes(1,2) # (num_traj, time_res, 3)
        else:
            if np.ndim(dif_obs)==2:
                T = len(self.obs_times)
                # T = sol.shape[1]
                dif_obs = dif_obs[:,None,:]
            elif np.ndim(dif_obs)==3:
                T = 1
            d_obs = dif_obs/self.nzvar[:,None,:] # (num_traj, 1, 3 or 9)
            if wrt=='obs_sol':
                g = d_obs
                if g.shape[1]==1: g=np.squeeze(g, axis=1)
            elif wrt=='sol':
                # g = np.tile(d_obs[:,:,:3]/T,(1,T,1)) # (num_traj, time_res, 3)
                g = np.tile(d_obs[:,:,:3]/T,(1,sol.shape[1],1)) # (num_traj, time_res, 3)
                if self.avg_traj=='aug':
                    g += d_obs[:,:,3:6]*sol*2/T
                    sol3 = np.zeros(sol.shape+(3,))
                    sol3[:,:,0,0] = sol[:,:,1]
                    sol3[:,:,0,1] = sol[:,:,0]
                    sol3[:,:,1,0] = sol[:,:,2]
                    sol3[:,:,1,2] = sol[:,:,0]
                    sol3[:,:,2,1] = sol[:,:,2]
                    sol3[:,:,2,2] = sol[:,:,1]
                    g += np.sum(d_obs[:,:,6:][:,:,:,None]*sol3/T,axis=2)
            else:
                g = None
        return g

if __name__ == '__main__':
    np.random.seed(2021)
    # define misfit
    num_traj=1; avg_traj='aug'
    mft = misfit(num_traj=num_traj, avg_traj=avg_traj, save_obs=False, STlik=False)
    
    # test gradient
    sol = mft.ode.solve(params=(10.0, 3.0, 28.0), t=mft.obs_times)
    c = mft.cost(sol)
    g = mft.grad(sol, 'sol')
    h = 1e-7
    dsol = np.random.randn(*sol.shape)
    c1 = mft.cost(sol +  h*dsol)
    gdsol_fd = (c1-c)/h
    gdsol = np.sum(g*dsol)
    rdiff_gdsol = np.linalg.norm(gdsol_fd-gdsol)/np.linalg.norm(dsol)
    print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gdsol)
    