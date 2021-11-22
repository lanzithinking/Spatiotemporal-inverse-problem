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
import matplotlib.pyplot as plt

class misfit:
    """
    Misfit for Lorenz63 inverse problem.
    """
    def __init__(self, obs_times=None, **kwargs):
        if obs_times is None:
            t_init = kwargs.get('t_init',0.)
            t_final = kwargs.get('t_final',4.)
            time_res = kwargs.get('time_res',100)
            self.obs_times = np.linspace(t_init, t_final, time_res)
        # get observations
        self.obs = self.get_obs()
        
        self.STlik = kwargs.pop('STlik',True)
        if self.STlik:
            self.stgp = kwargs.get('stgp')
            if self.stgp is None:
                # define STGP kernel for the likelihood (misfit)
                # self.stgp=STGP(spat=self.targets, temp=self.observation_times, opt=kwargs.pop('ker_opt',0), jit=1e-2)
                C_x=GP(self.targets, l=.5, sigma2=np.sqrt(self.noise_variance), store_eig=True, jit=1e-2)
                C_t=GP(self.observation_times, store_eig=True, l=.2, sigma2=np.sqrt(self.noise_variance))#, ker_opt='matern',nu=.5)
                # C_x=GP(self.targets, l=.4, jit=1e-3, sigma2=.1, store_eig=True)
                # C_t=GP(self.observation_times, store_eig=True, l=.2, sigma2=.1, ker_opt='matern',nu=.5)
                self.stgp=STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',0), spdapx=False)
                # C_x=GP(self.targets, l=.5, sigma2=.1, store_eig=True)
                # C_t=GP(self.observation_times, store_eig=True, l=.2, sigma2=1.)#, ker_opt='matern',nu=.5)
                # self.stgp=STGP_mg(STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',2), spdapx=False), K=1, nz_var=self.noise_variance, store_eig=True)
    
    def get_obs(self, avg_traj=True, **kwargs):
        """
        Obtain observations
        """
        fld=kwargs.pop('obs_file_loc',os.getcwd())
        try:
            f=open(os.path.join(fld,'Lorenz_obs.pckl'),'rb')
            obs=pickle.load(f)
            f.close()
            print('Observation file has been read!')
        except Exception as e:
            print(e)
            num_traj=kwargs.pop('num_traj',10)
            init=kwargs.pop('init',-15 + 30 * np.random.random((num_traj, 3)))
            ode=kwargs.pop('ode',ode(x0=init))
            obs=ode.solve(t=self.obs_times)
            if kwargs.pop('save_obs',True):
                f=open(os.path.join(fld,'Lorenz_obs.pckl'),'wb')
                pickle.dump(obs,f)
                f.close()
        if avg_traj:
            obs =  obs.mean(axis=1)
        return obs
    
    def observe(self, t=None):
        """
        Observation operator
        """
        
    
    def cost(self, u, option='nll'):
        """
        Compute misfit
        option: return negative loglike ('nll') or (postive) quadratic form (quad) where loglike = halfdelt+quad
        """
        
    def grad(self, u):
        """
        Compute the gradient of misfit
        """
        

if __name__ == '__main__':
    np.random.seed(2021)
    