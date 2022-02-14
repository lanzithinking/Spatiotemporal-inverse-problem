#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The misfit of Rossler inverse problem 
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Shuyi Li"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.4"
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
    Misfit for lorenz96 inverse problem.
    """
    def __init__(self, ode=None, obs_times=None, **kwargs):
        """
        ode: Rossler object
        obs_times: time points where observations are taken
        true_params: true parameters of Rossler for which observations are obtained
        avg_traj: whether to average the solution along each of the trajectories - False (whole trajectories); True (average trajectories); 'aug' (augment averaged trajectories with 2nd order interactions)
        STlik: whether to use spatiotemporal models; has to be used with 'avg_traj=False'
        stgp: spatiotemporal GP kernel
        """
        self.ode = lorenz96(**kwargs) if ode is None else ode
        if obs_times is None:
            t_init = kwargs.pop('t_init',0.)
            t_final = kwargs.pop('t_final',10.)
            time_res = kwargs.pop('time_res',200)
            self.obs_times = np.linspace(t_init, t_final, time_res)
        else:
            self.obs_times = obs_times
        self.true_params = kwargs.pop('true_params',{'h':self.ode.h, 'F':self.ode.F, 'logc':self.ode.logc, 'b':self.ode.b})
        self.avg_traj = kwargs.pop('avg_traj',True) # False, True, or 'aug'
        # get observations
        self.obs, self.nzvar = self.get_obs(**kwargs)
        print(self.obs.shape)
        self.STlik = kwargs.pop('STlik',False)
        if self.STlik:
            if self.avg_traj:
                warnings.warn('Spatiotemporal models not available for time-averaged trajectories!')
                self.avg_traj = False
            self.stgp = kwargs.get('stgp')
            if self.stgp is None:
                # define STGP kernel for the likelihood (misfit)
                sigma2_ = self.nzvar.mean(axis=0)
                if np.ndim(self.nzvar)==3: sigma2_ = np.diag(sigma2_)
                if self.STlik in (True,'sep'):
                    # -- model 0 -- separable STGP
                    # self.stgp=STGP(spat=self.obs.mean(axis=0).T, temp=self.obs_times, opt=kwargs.pop('ker_opt',0), jit=1e-2)
                    # C_x=GP(self.obs.mean(axis=0).T, l=.3, sigma2=np.diag(np.sqrt(sigma2_)), store_eig=True)
                    l_=[.1, 1e-2, 1e-2, 1e-3, 1e-4]*self.ode.K
                    self.l = np.zeros(int(5*self.ode.K*(5*self.ode.K-1)/2))
                    k = 0
                    for i in range(5*self.ode.K-1):
                        for j in range(i+1,5*self.ode.K):
                            self.l[k] = np.sqrt(l_[i]*l_[j])
                            k+=1
                    sigma2=np.ones((self.obs.shape[1],)*2); np.fill_diagonal(sigma2, np.sqrt(sigma2_))
                    C_x=GP(self.obs.T, l=self.l, sigma2=sigma2, store_eig=True)
                    C_t=GP(self.obs_times, store_eig=True, l=1.0, sigma2=np.sqrt(sigma2_.sum()), jit=1e-2)#, ker_opt='matern',nu=.5)
                    self.stgp=STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',0), spdapx=False)
                    # self.stgp=STGP_mg(STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',2), spdapx=False), K=1, nz_var=self.nzvar.mean(axis=0).sum(), store_eig=True)
                elif self.STlik=='full':
                    # -- model 1 -- full STGP
                    sigma2=np.ones((np.prod(self.obs.shape[:]),)*2); np.fill_diagonal(sigma2, np.sqrt(np.tile(sigma2_,len(self.obs_times))))
                    self.stgp=GP(self.obs.flatten(), l=.3, sigma2=sigma2, store_eig=True, jit=1e-2)
                else:
                    raise NotImplementedError('Model not implemented!')
    
    def get_obs(self, **kwargs):
        """
        Obtain observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        var_out=kwargs.pop('var_out','cov') # False, True, or 'cov
        obs_file_name='Lorenz96_obs_'+{True:'avg',False:'full','aug':'avgaug'}[self.avg_traj]+'_traj_'+{True:'nzvar',False:'','cov':'nzcov'}[var_out]+'.pckl'
        use_saved_obs=kwargs.pop('use_saved_obs',True)
        if use_saved_obs and os.path.exists(os.path.join(obs_file_loc,obs_file_name)):
            f=open(os.path.join(obs_file_loc,obs_file_name),'rb')
            obs, nzvar=pickle.load(f)
            f.close()
            print('Observation file '+obs_file_name+' has been read!')
        else:
            ode=kwargs.pop('ode',self.ode)
            params=kwargs.pop('params',tuple(self.true_params.values()))
            t=self.obs_times
            sol=ode.solve(params=params, t=t)
            obs, nzvar=self.observe(sol, var_out=var_out)
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                f=open(os.path.join(obs_file_loc,obs_file_name),'wb')
                print(os.path.join(obs_file_loc,obs_file_name))
                pickle.dump([obs, nzvar], f)
                f.close()
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzvar
    
    def observe(self, sol=None, var_out=False):
        """
        Observation operator
        --------------------
        var_out: option to output variance - False (no variances); True (only variance); 'cov' (whole covariance matrix)
        """
        if sol is None:
            sol = self.ode.solve(params=tuple(self.true_params.values()), t=self.obs_times)
        num_traj = 1
        #sol[0]--xts:(time_res, K), sol[1]--yts:(time_res, K, L)
        #assert np.ndim(sol)==3, 'The shape of solution should be (num_traj, time_res, 3)!'
        xts, yts = sol
        ykmean = yts.mean(axis=2)
        obs = np.zeros((len(self.obs_times),self.ode.K, 5)) #(time_res, K, 5)
        for i in range(len(self.obs_times)):
            xts_i = xts[i][:,np.newaxis]
            ykmean_i = ykmean[i][:,np.newaxis]
            obs[i] = np.hstack((xts_i, ykmean_i, xts_i**2, xts_i*ykmean_i, ykmean_i**2)) #(K,5)
        #(time_res, K*5)
        obs = obs.reshape((len(self.obs_times), -1))
        # if var_out: nzvar = sol.var(axis=1)
        
        if self.avg_traj:
            obsf = obs.mean(axis=0) # (K, 5)/(K*5)
        else:
            obsf = obs # (time_res, K, 5)/(time_res, K*5)
        if var_out:
            nzvar = []
            for i in range(num_traj):                
                solxt_i = obs
            if var_out=='cov':
                nzvar.append(np.cov(solxt_i, rowvar=False))
            else:
                nzvar.append(solxt_i.var(axis=0))
            nzvar = np.array(nzvar) #(1*5K*5K) if cov else (1*5K)
        return (obsf, nzvar) if var_out else obsf
    
    def cost(self, sol=None, obs=None, option='nll'):
        """
        Compute misfit
        option: return negative loglike ('nll') or (postive) quadratic form (quad) where loglike = halfdelt+quad
        """
        if obs is None:
            if sol is None:
                sol = self.ode.solve(params=tuple(self.true_params.values()), t=self.obs_times)
            obs = self.observe(sol=sol)
        dif_obs = (obs - self.obs)[np.newaxis,:] #(num_trial(1), time_res, 5*K), (num_trial(1), 5*K)
        if self.STlik:
            logpdf,half_ldet = self.stgp.matn0pdf(np.swapaxes(dif_obs,0,2).reshape((-1,dif_obs.shape[0]),order='F'))
            res = {'nll':-logpdf, 'quad':-(logpdf - half_ldet), 'both':[-logpdf,-(logpdf - half_ldet)]}[option]
        else:
            if np.ndim(dif_obs)==3: dif_obs = np.swapaxes(dif_obs, 1,2)
            if np.ndim(self.nzvar)==3:
                res = np.sum(dif_obs*np.linalg.solve(self.nzvar,dif_obs))/2
            elif np.ndim(self.nzvar)==2:
                if np.ndim(dif_obs)==3: self.nzvar = self.nzvar[:,:,None]
                res = np.sum(dif_obs**2/self.nzvar)/2
        return res
        
    
    def sample(self):
        """
        Sample noise
        """
        num_traj = self.obs.shape[0]
        if self.STlik:
            noise = self.stgp.sample_priM(num_traj).reshape(np.array(self.obs.shape)[[2,1,0]]).swapaxes(0,2)
        else:
            noise = []
            for nzvar_i in self.nzvar:
                noise.append(np.sqrt(nzvar_i)*np.random.randn(len(nzvar_i)) if np.ndim(self.nzvar)==2 else 
                             np.random.multivariate_normal(np.zeros(nzvar_i.shape[1]), nzvar_i))
            noise = np.stack(noise)
        return noise
    

if __name__ == '__main__':
            
    np.random.seed(2021)
    num_traj = 1
    t = np.linspace(0, 10, 200)
    L, K = 2, 2
    n = (L+1) * K 
    x0 = np.random.randn(num_traj,n)
    
    ode = lorenz96(x0=x0, t=t, L=L, K=K)
    
    # define misfit
    avg_traj=False; var_out='cov'; STlik=True
    
    mft = misfit(ode=ode, obs_times=t, num_traj=num_traj, avg_traj=avg_traj, var_out=var_out, save_obs=False, STlik=STlik)
    
    # test gradient
    sol = mft.ode.solve(params=(1, 10, np.log(10), 10), t=mft.obs_times)
    c = mft.cost(sol)
#    g = mft.grad(sol, wrt='sol')
#    h = 1e-7
#    dsol = np.random.randn(*sol.shape)
#    c1 = mft.cost(sol +  h*dsol)
#    gdsol_fd = (c1-c)/h
#    gdsol = np.sum(g*dsol)
#    rdiff_gdsol = np.linalg.norm(gdsol_fd-gdsol)/np.linalg.norm(dsol)
#    print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gdsol)
#    