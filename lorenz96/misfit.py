#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The misfit of Lorenz96 inverse problem 
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Shuyi Li"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.5"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

'''
# import modules
import numpy as np
from scipy import integrate
from scipy.spatial.distance import pdist,squareform
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
    Misfit for Lorenz96 inverse problem.
    """
    def __init__(self, ode=None, obs_times=None, **kwargs):
        """
        ode: Lorenz96 object
        obs_times: time points where observations are taken
        true_params: true parameters of Lorenz96 for which observations are obtained
        avg_traj: whether to average the solution along each of the trajectories - False (whole trajectories); True (average trajectories); 'aug' (augment averaged trajectories with 2nd order interactions)
        STlik: whether to use spatiotemporal models; has to be used with 'avg_traj=False'
        stgp: spatiotemporal GP kernel
        """
        self.ode = lrz96(**kwargs) if ode is None else ode
        if obs_times is None:
            t_init = kwargs.pop('t_init',100.)
            t_final = kwargs.pop('t_final',110.)
            time_res = kwargs.pop('time_res',100)
            self.obs_times = np.linspace(t_init, t_final, time_res)
        else:
            self.obs_times = obs_times
        self.true_params = kwargs.pop('true_params',{'h':self.ode.h, 'F':self.ode.F, 'logc':self.ode.logc, 'b':self.ode.b})
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
                sigma2_ = self.nzvar.mean(axis=0)
                if np.ndim(self.nzvar)==3: sigma2_ = np.diag(sigma2_)
                # if np.ndim(self.nzvar)==2: sigma2_ = np.diag(sigma2_)
                if self.STlik in (True,'sep'):
                    # -- model 0 -- separable STGP
                    # self.stgp=STGP(spat=self.obs.mean(axis=0).T, temp=self.obs_times, opt=kwargs.pop('ker_opt',0), jit=1e-2)
                    # C_x=GP(self.obs.mean(axis=0).T, l=.3, sigma2=np.diag(np.sqrt(sigma2_)), store_eig=True)
                    sigma2=np.ones((self.obs.shape[2],)*2); np.fill_diagonal(sigma2, np.sqrt(sigma2_))
                    # C_x=GP(self.obs.mean(axis=0).T, l=.4, sigma2=sigma2, store_eig=True)
                    # l_=np.repeat([.1, 1e-2, 1e-2, 1e-3, 1e-4],self.ode.K)
                    # l_x = np.ones(int(5*self.ode.K*(5*self.ode.K-1)/2))
                    # k = 0
                    # for i in range(5*self.ode.K-1):
                    #     for j in range(i+1,5*self.ode.K):
                    #         l_x[k] = np.sqrt(l_[i]*l_[j])
                    #         k+=1
                    # l_=np.repeat([.1, 1e-2],self.ode.K)
                    # l_x = squareform(np.sqrt(self.ode.K*l_[:,None]*l_),checks=False)
                    l_=np.repeat([1, .1],self.ode.K)
                    l_x = squareform(self.ode.K*l_[:,None]*l_,checks=False)
                    C_x=GP(self.obs.mean(axis=0).T, l=l_x, sigma2=sigma2, store_eig=True)
                    C_t=GP(self.obs_times, store_eig=True, l=0.1, sigma2=np.sqrt(sigma2_.sum()), ker_opt='matern',nu=1.5)
                    # C_x=GP(self.obs.mean(axis=0).T); C_x.tomat=lambda : sigma2_
                    # C_t=GP(self.obs_times, store_eig=True, l=0.1, sigma2=1.0, ker_opt='matern',nu=0.5)
                    self.stgp=STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',0), spdapx=False)
                    # self.stgp=STGP_mg(STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',2), spdapx=False), K=1, nz_var=self.nzvar.mean(axis=0).sum(), store_eig=True)
                elif self.STlik=='full':
                    # -- model 1 -- full STGP
                    sigma2=np.ones((np.prod(self.obs.shape[1:]),)*2); np.fill_diagonal(sigma2, np.sqrt(np.tile(sigma2_,len(self.obs_times))))
                    # self.stgp=GP(self.obs.flatten(), l=.4, sigma2=sigma2, store_eig=True, jit=1e-2)
                    l_=np.tile(np.repeat([.1, 1e-2],self.ode.K), len(self.obs_times))
                    l_x = squareform(np.sqrt(l_[:,None]*l_),checks=False)
                    self.stgp=GP(self.obs.flatten(), l=l_x, sigma2=sigma2, store_eig=True, jit=1e-2)
                else:
                    raise NotImplementedError('Model not implemented!')
    
    def get_obs(self, **kwargs):
        """
        Obtain observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        var_out=kwargs.pop('var_out','cov') # False, True, or 'cov
        obs_file_name='Lorenz_obs_'+{True:'avg',False:'full','aug':'avgaug'}[self.avg_traj]+'_traj_'+{True:'nzvar',False:'','cov':'nzcov'}[var_out]+'.pckl'
        use_saved_obs=kwargs.pop('use_saved_obs',True)
        if use_saved_obs:
            try:
                f=open(os.path.join(obs_file_loc,obs_file_name),'rb')
                loaded=pickle.load(f)
                obs, nzvar=loaded[:2]
                if len(loaded)==3:
                    times=loaded[2]
                    d=abs(self.obs_times-times[:,None])
                    obs=obs[:,np.argmin(d,axis=0),:]
                    dt=np.ptp(self.obs_times)/len(self.obs_times)
                    msk=(np.min(d,axis=0)<dt)
                    if not np.any(msk):
                        raise ValueError('No time series found!')
                    if not np.all(msk):
                        self.obs_times=self.obs_times[msk]
                        obs=obs[:,msk,:]
                        print('Partial time series found!')
                f.close()
                print('Observation file '+obs_file_name+' has been read!')
            except Exception as e:
                print(e); pass
                use_saved_obs=False
        if not use_saved_obs:
            ode=kwargs.pop('ode',self.ode)
            params=kwargs.pop('params',tuple(self.true_params.values()))
            t=self.obs_times
            sol=ode.solve(params=params, t=t)
            obs, nzvar=self.observe(sol, var_out=var_out)
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                f=open(os.path.join(obs_file_loc,obs_file_name),'wb')
                dump_list=[obs, nzvar]
                if not self.avg_traj: dump_list.append(self.obs_times)
                pickle.dump(dump_list, f)
                f.close()
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzvar
    
    def _split(self, sol):
        K,L = self.ode.K,self.ode.L
        X = sol[:,:,:K]; Y = sol[:,:,K:].reshape(sol.shape[:2]+(L,K))
        return X,Y
    
    def _condenseY(self, sol):
        X,Y = self._split(sol)
        sol_cnds = np.concatenate((X,Y.mean(axis=2)),axis=-1)
        return sol_cnds
    
    def observe(self, sol=None, var_out=False):
        """
        Observation operator
        --------------------
        var_out: option to output variance - False (no variances); True (only variance); 'cov' (whole covariance matrix)
        """
        if sol is None:
            sol = self.ode.solve(params=tuple(self.true_params.values()), t=self.obs_times)
        assert np.ndim(sol)==3, 'The shape of solution should be (num_traj, time_res, K(1+L))!'
        num_traj = sol.shape[0]
        X,Y = self._split(sol)
        sol = np.concatenate((X,Y.mean(axis=2)),axis=-1) # (num_traj, time_res, 2K)
        if self.avg_traj:
            obs = sol.mean(axis=1) # (num_traj, 2K)
            if self.avg_traj=='aug':
                obs = np.hstack([obs,np.concatenate([X**2,np.mean(Y**2,axis=2),X*Y.mean(axis=2)],axis=-1).mean(axis=1)]) # (num_traj, 5K)
        else:
            obs = sol # (num_traj, time_res, 2K)
        if var_out:
            nzvar = []
            for i in range(num_traj):
                if self.avg_traj=='aug':
                    solxt_i = np.hstack([sol[i],X[i]**2,np.mean(Y[i]**2,axis=1),X[i]*Y[i].mean(axis=1)])
                else:
                    solxt_i = sol[i]
                if var_out=='cov':
                    nzvar.append(np.cov(solxt_i, rowvar=False))
                else:
                    nzvar.append(solxt_i.var(axis=0))
            nzvar = np.array(nzvar)
        return (obs, nzvar) if var_out else obs
    
    def cost(self, sol=None, obs=None, option='nll'):
        """
        Compute misfit
        option: return negative loglike ('nll') or (postive) quadratic form (quad) where loglike = halfdelt+quad
        """
        if obs is None:
            if sol is None:
                sol = self.ode.solve(params=tuple(self.true_params.values()), t=self.obs_times)
            obs = self.observe(sol=sol)
        dif_obs = obs - self.obs
        if self.STlik:
            logpdf,half_ldet = self.stgp.matn0pdf(np.swapaxes(dif_obs,0,2).reshape((-1,dif_obs.shape[0]),order='F'))
            res = {'nll':-logpdf, 'quad':-(logpdf - half_ldet), 'both':[-logpdf,-(logpdf - half_ldet)]}[option]
        else:
            if np.ndim(dif_obs)==3: dif_obs = np.swapaxes(dif_obs, 1,2)
            if np.ndim(self.nzvar)==3:
                res = np.sum(dif_obs*np.linalg.solve(self.nzvar,dif_obs))/2
            elif np.ndim(self.nzvar)==2:
                nzvar = self.nzvar[:,:,None] if np.ndim(dif_obs)==3 else self.nzvar
                res = np.sum(dif_obs**2/nzvar)/2
        return res
    
    def grad(self, sol=None, obs=None, wrt='obs'):
        """
        Compute the gradient of misfit
        """
        if obs is None:
            if sol is None:
                sol = self.ode.solve(params=tuple(self.true_params.values()), t=self.obs_times)
            obs = self.observe(sol=sol)
        dif_obs = obs - self.obs
        if self.STlik:
            g = self.stgp.solve(np.swapaxes(dif_obs,0,2).reshape((-1,dif_obs.shape[0]),order='F')).reshape(np.array(dif_obs.shape)[[2,1,0]],order='F').swapaxes(0,2) # (num_traj, time_res, 2K)
        else:
            if np.ndim(dif_obs)==2:
                T = len(self.obs_times)
                # T = sol.shape[1]
                if np.ndim(self.nzvar)==3:
                    d_obs = np.linalg.solve(self.nzvar, dif_obs)
                elif np.ndim(self.nzvar)==2:
                    d_obs = dif_obs/self.nzvar
                d_obs = d_obs[:,None,:]
            elif np.ndim(dif_obs)==3:
                T = 1
                if np.ndim(self.nzvar)==3:
                    d_obs = np.swapaxes(np.linalg.solve(self.nzvar, np.swapaxes(dif_obs, 1,2)), 1,2)
                elif np.ndim(self.nzvar)==2:
                    d_obs = dif_obs/self.nzvar[:,None,:]
            if wrt=='obs':
                g = d_obs
                if g.shape[1]==1: g=np.squeeze(g, axis=1)
            elif wrt=='sol':
                K=self.ode.K
                sol = self._condenseY(sol)
                # g = np.tile(d_obs[:,:,:2*K]/T,(1,T,1)) # (num_traj, time_res, 2K)
                g = np.tile(d_obs[:,:,:2*K]/T,(1,sol.shape[1],1)) # (num_traj, time_res, 2K)
                if self.avg_traj=='aug':
                    g += d_obs[:,:,2*K:4*K]*sol*2/T # not exact for g[:,:,3K:4K] for d\bar{Y^2}/d\bar{Y}(t); but gradient is not used in adjoint which is not implemented
                    sol3 = np.zeros(sol.shape+(K,))
                    for k in range(K):
                        sol3[:,:,k,k] = sol[:,:,K+k]
                        sol3[:,:,K+k,k] = sol[:,:,k]
                    g += np.sum(d_obs[:,:,4*K:][:,:,None,:]*sol3/T,axis=-1)
            else:
                g = None
        return g
    
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
    # define misfit
    num_traj=1; avg_traj=False; var_out=True
    mft = misfit(num_traj=num_traj, avg_traj=avg_traj, var_out=var_out, save_obs=False, STlik=True)
    
    # test gradient
    sol = mft.ode.solve(params=(1.0, 10.0, np.log(10), 10.0), t=mft.obs_times)
    c = mft.cost(sol)
    g = mft.grad(sol, wrt='sol')
    h = 1e-4
    dsol = np.random.randn(*sol.shape)
    c1 = mft.cost(sol +  h*dsol)
    gdsol_fd = (c1-c)/h
    gdsol = np.sum(g*mft._condenseY(dsol))
    rdiff_gdsol = np.linalg.norm(gdsol_fd-gdsol)/np.linalg.norm(dsol)
    print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gdsol)
    