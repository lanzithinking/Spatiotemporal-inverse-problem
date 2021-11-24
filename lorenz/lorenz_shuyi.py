#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 18:13:35 2021

@author: apple
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
The Lorenz system of differential equations written in Scipy
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Shuyi Li"
__copyright__ = "Copyright 2020, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

1: get observation(x,y,z) given t and u 
2: sample u misfit: Obs-G(u) where Obs=(x,y,z)

'''
# import modules
import numpy as np
from scipy import integrate
import scipy.stats as stats
import matplotlib.pyplot as plt

import sys
#import os
sys.path.append( "../" )
from sampler.slice import slice as slice_sampler
from util.stgp.GP import GP
from util.stgp.STGP import STGP
#from util.stgp.STGP_mg import STGP_mg

def solve_lorenz(x0=None, t=None, N=10, t0=0, max_time=4.0, sigma=10.0, beta=8./3, rho=28.0):
    # Choose random starting points, uniformly distributed from -15 to 15
    #np.random.seed(1)
    if x0 is None:
        x0 = -15 + 30 * np.random.random((N, 3))
    if t is None:
        t = np.linspace(t0, max_time, int(250*max_time))
        
    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        """Compute the time-derivative of a Lorenz system."""
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    # Solve for the trajectories
    
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0i, t)
                      for x0i in x0])

    return t, x_t


## generate prior defined in CES
def gene_prior():
    # for beta and r
    mu = np.array([[1.2],[3.3]])
    std = np.diag((.5, .15))
    z = np.random.randn(2,1)
    u = np.dot(std, z) + mu
    
    return u.squeeze()


class Lorenz:
    def __init__(self, observation_times, x0, obs=None, augment = True, STlik=False, **kwargs):
        """
        Initialize the lorenz 63 problem by defining the ODE model, the prior model and the misfit (likelihood) model.
        y ~ MN(G(u),C), G(u) forward evaluation u refers to hyperpatater in lorenz(sigma, beta, rho), G(u)->x,y,z obs
        
        observation_times: time interval we choose to observe
        x0: initial conditions for space location x,y,z
        obs: observation number of trail*number of time*space dimension = n*t*3 for spatiotemporal case, 
            n*3/n*9 for time averaged data when STlik=False
        augment: whether change original 3(x,y,z)->9(x,y,z,x**2,y**2,z**2,xy,yz,xz)
        STlik: whether for spatiotemporal covariance kernal
        """
        
        self.obs=obs
        self.observation_times = observation_times
        self.x0 = x0
        self.augment = augment
        self.STlik = STlik
        if self.STlik:
            self.augment = False #must be false when using spatiotemporal likelihood
            
        
        if self.obs is None:
            # solve with real beta & rho
            G_u, timeaveG_u = self.solve([np.log(8/3), np.log(28)])
            #3*t*trails for STlik=T, consistent with Cx*Ct
            self.obs = G_u.transpose((2,1,0)) if self.STlik else timeaveG_u
        #calculate emperical Gamma, covariance matrix 
        if self.x0.shape[0] == 1 and not self.STlik:
            raise Exception("Sorry, no emperical covariance matrix for only 1 trial")
        self.noise_variance = kwargs.pop('noise_variance', np.cov(self.obs.T) if not self.STlik 
                                         else np.var(self.obs)) #np.diag
        
        
        if self.STlik:
             
            C_x=GP(timeaveG_u, l=.5, sigma2=np.sqrt(self.noise_variance), store_eig=True, jit=1e-2)
            C_t=GP(self.observation_times, store_eig=True, l=.2, sigma2=np.sqrt(self.noise_variance))
            self.stgp=STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',0), spdapx=False)
    
    def solve_lorenz(self, t=None, sigma=10.0, beta=8./3, rho=28.0):
        """
        forward evaluation to obtain G(u) = x,y,z
        """
        
        if t is None:
            t =self.observation_times
            
        def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
            """Compute the time-derivative of a Lorenz system."""
            x, y, z = x_y_z
            return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
        # Solve for the trajectories
        
        x_t = np.asarray([integrate.odeint(lorenz_deriv, x0i, t)
                          for x0i in self.x0])
    
        return t, x_t  
    
    def solve(self, logu, t=None):
        """
        obtain du = G(u)-obs
        """
        
        #of trial N * t * 3
        _, G_u = self.solve_lorenz(t, sigma=10.0, beta=np.exp(logu[0]), rho=np.exp(logu[1]))   
        # N * 3, convert to 1 dimensional for GP(3,)
        timeaveG_u = G_u.mean(axis=1).squeeze() if self.STlik else G_u.mean(axis=1)
        
        if self.augment:
            G_u2 = (G_u**2).mean(axis=1)
            xy = (G_u[:,:,0]*G_u[:,:,1]).mean(axis=1, keepdims = True)
            yz = (G_u[:,:,1]*G_u[:,:,2]).mean(axis=1, keepdims = True)
            xz = (G_u[:,:,0]*G_u[:,:,2]).mean(axis=1, keepdims = True)
            timeaveG_u = np.hstack((timeaveG_u,G_u2,xy,yz,xz))
        
        return G_u, timeaveG_u
    
    def misfit(self, logu, option='ll'):
        """
        du: obs-G(u)
        noise_variance: likelihood Y ~ N(G(u), noise_variance)
        return: loglik
        """
        
        G_u, timeaveG_u = self.solve(logu)
        
        if self.STlik:
            G_u = G_u.transpose((2,1,0))
            du = G_u - self.obs
            logpdf,half_ldet = self.stgp.matn0pdf(du)
            res = {'ll':logpdf, 'quad':-(logpdf - half_ldet), 'both':[logpdf,-(logpdf - half_ldet)]}[option]
        else:
            du = timeaveG_u - self.obs 
            res = -np.sum(np.dot(du, np.linalg.inv(self.noise_variance))*du)
            #-np.sum( np.dot(np.dot(du[i], np.linalg.inv(cov)), du[i] ) for i in range(du.shape[0]) )
            #res = -np.sum(du*du/(2.*self.noise_variance) )
            
        return res
    
    def pdf_calc(self, logu, mean=np.array([1.2, 3.3]), cov=np.diag((.5**2,.15**2))):
        """
        logprior
        """
        return np.log(stats.multivariate_normal.pdf(logu, mean=mean , cov=cov) )

    def get_pospdf(self, logbeta, logrho):
        logu = [logbeta, logrho]
        prior =  self.pdf_calc(logu, np.array([1.2, 3.3]), np.diag((.5**2,.15**2)))
        lik = self.misfit(logu)
        
        return prior + lik
    
    def MH(self, logu, TARGET_SIGMA=None):
    
        #metropolis hasting
        
        if TARGET_SIGMA is None:
            TARGET_SIGMA = np.diag((.25,.25))
        u_init = logu
        
        u_prop = np.random.multivariate_normal(mean=u_init, cov=TARGET_SIGMA)
        
        prior_init = self.pdf_calc(u_init, np.array([1.2, 3.3]), np.diag((.5**2,.15**2)))
        prior_prop = self.pdf_calc(u_prop, np.array([1.2, 3.3]), np.diag((.5**2,.15**2)))
        
        like_init = self.misfit(u_init)
        like_prop = self.misfit(u_prop)
        
        #we use gaussian symmetric, so get rid of this
        #target_init = pdf_calc(u_prop, u_init, TARGET_SIGMA)
        #target_prop = pdf_calc(u_init, u_prop, TARGET_SIGMA)
    
        log_accept_prob = (prior_prop + like_prop) - \
                        (prior_init + like_init)
    
        random = np.random.uniform(low=0, high=1)
        log_random = np.log(random)
        
        if log_random < log_accept_prob:
    
            u_init = u_prop
            acpt = True
        else:
            acpt = False
    
        return u_init, acpt

    
    def pos_trace(self, logu, TARGET_SIGMA=None, CHAIN_LEN=5000, useslice=True):
        # np.random.seed(0)
        if TARGET_SIGMA is None:
            #step size
            TARGET_SIGMA = np.diag((.25,.25))
       
        #beta, rho
        u_list = np.zeros((CHAIN_LEN,2))
        count=0
        for epoch in range(CHAIN_LEN):
            
            if useslice:
                
                logbeta,_ = slice_sampler(logu[0], self.get_pospdf(logu[0],logu[1]), logf= lambda u: self.get_pospdf(u,logu[1]))
                logu = [logbeta,logu[1]]
                ####################check
                #print(logu)
                logrho,_ = slice_sampler(logu[1], self.get_pospdf(logu[0],logu[1]), logf= lambda u: self.get_pospdf(logu[0],u))
                logu = [logu[0],logrho]
                ####################check
                #print(logu)
            else:
                logu,acpt = self.MH(logu, TARGET_SIGMA=TARGET_SIGMA)
                count += acpt
            
            u_list[epoch] = logu
            
        if not useslice:
            print('Acceptance ratio is: {}'.format(count/CHAIN_LEN))
    
        return u_list,count/CHAIN_LEN
    
        


if __name__ == '__main__':
    np.random.seed(2021)
    d = 3
    # if change obs from (x,y,z) to (x,y,z,x**2,y**2,z**2,xy,yz,xz) CES paper 9*9
    # (g(u)-obs)cov^(-1)(g(u)-obs) 10*9
    augment = True
    N = 1
    x0 = -15 + 30 * np.random.random((N, d))   
    
    time_resolution = 40
    negini = 0
    t_1 = 0
    t_final = 20
    observation_times = np.linspace(t_1, t_final, num = time_resolution*t_final+1)
    
    #construct lorenz problem
    lorenz = Lorenz(observation_times[negini:], x0, obs=None, augment = augment, STlik=True)
    
    #check for forward evaluation G(u)
    #_,check = lorenz.solve_lorenz(beta=np.exp(np.log(8/3)),rho=np.exp(np.log(28))) 
    #check for misfit
    #lorenz.misfit(logu=logu)
    logu = gene_prior()
    
    #we don't need acceptance ratio for slice
    u_sampleslice,_ = lorenz.pos_trace(logu, CHAIN_LEN=1000, useslice=True)
    u_samplemh,ar = lorenz.pos_trace(logu, TARGET_SIGMA = np.diag((.005,.01)), CHAIN_LEN=1000, useslice=False)
    
    
    pos_u = np.median(np.exp(u_sample[:]),axis=0)
    _,G_u = solve_lorenz(x0=x0, t=observation_times, sigma=10.0, beta=pos_u[0], rho=pos_u[1])
    timeaveG_u = G_u.mean(axis=1)
    
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import cnames
    from matplotlib import animation
    
    def plot(obs):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.axis('off')
        
        # prepare the axes limits
        ax.set_xlim((-25, 25))
        ax.set_ylim((-35, 35))
        ax.set_zlim((5, 55))
        
        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, N))
        
        for i in range(N):
            x, y, z = obs[i,10:,:].T
            lines = ax.plot(x, y, z, '-', c=colors[i])
            plt.setp(lines, linewidth=2)
        
        ax.view_init(30)#azim=0
        plt.show()
        
    plot(G_u)
    
    #check posterior samples trace plot
    fig,axes = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=True,figsize=(16,4))
    for i, ax in enumerate(axes.flat):
        plt.axes(ax)
        plt.plot(np.exp(u_sample[:,0]),'o')
        plt.plot(np.exp(u_sample[:,1]),'o')
        #plt.title('Average $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
    
    #check trajectories
    fig,axes = plt.subplots(nrows=3,ncols=1,sharex=True,sharey=False,figsize=(16,10))
    for i, ax in enumerate(axes.flat):
        plt.axes(ax)
        plt.plot(observation_times[negini:][:], G_u[:,:,i].T)
        plt.title('Trajectories of $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
    #plt.savefig('trajectories.png',bbox_inches='tight')

        

    