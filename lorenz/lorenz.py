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
import os

def solve_lorenz(x0=None, t=None, N=10, max_time=4.0, sigma=10.0, beta=8./3, rho=28.0):
    # Choose random starting points, uniformly distributed from -15 to 15
    np.random.seed(1)
    if x0 is None:
        x0 = -15 + 30 * np.random.random((N, 3))
    if t is None:
        t = np.linspace(0, max_time, int(250*max_time))
        
    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):
        """Compute the time-derivative of a Lorenz system."""
        x, y, z = x_y_z
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    # Solve for the trajectories
    
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0i, t)
                      for x0i in x0])

    return t, x_t


def lorenz_deriv(x_y_z, t0, sigma=10, beta=8/3, rho=28):
    """Compute the time-derivative of a Lorenz system."""
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

## generate prior defined in CES
def gene_prior():
    # for beta and r
    mu = np.array([[1.2],[3.3]])
    std = np.diag((.5, .15))
    z = np.random.randn(2,1)
    u = np.dot(std, z) + mu
    
    return u.squeeze()

## likelihood
def misfit(u, obs, observation_times, x0, noise_variance=.01, STlik = False, augment=True):
    """
    u: beta, rho with sigma fixed = 10, 2 dimensional (same as CES paper)
    obs: observation by solving real ODE
    observation_times: time interval
    x0: initial values x,y,z
    noise_variance: likelihood Y ~ N(G(u), noise_variance)
    """
    #of trial N * t * 3
    _, G_u = solve_lorenz(x0=x0, t=observation_times, sigma=10.0, beta=np.exp(u[0]), rho=np.exp(u[1]))
   
    timeaveG_u = G_u.mean(axis=1) # #of trial N * 3
    if augment:
        G_u2 = (G_u**2).mean(axis=1)
        xy = (G_u[:,:,0]*G_u[:,:,1]).mean(axis=1, keepdims = True)
        yz = (G_u[:,:,1]*G_u[:,:,2]).mean(axis=1, keepdims = True)
        xz = (G_u[:,:,0]*G_u[:,:,2]).mean(axis=1, keepdims = True)
        timeaveG_u = np.hstack((timeaveG_u,G_u2,xy,yz,xz))
    if STlik:
        du = timeaveG_u - obs       
        logpdf,half_ldet = self.stgp.matn0pdf(du)
        res = {'nll':-logpdf, 'quad':-(logpdf - half_ldet), 'both':[-logpdf,-(logpdf - half_ldet)]}[option]
    else:
        
        du = timeaveG_u - obs 
        c = np.sum(du*du)
        res = c/(2.*noise_variance)
        
    return res


def target_draw(logu, sigma=np.diag((.25,.25))):
    draw = lambda : np.random.multivariate_normal(mean=logu, cov=sigma, size=1)[0]
    return draw()

#here prior and transition with same multivariate gaussian 
def pdf_calc(logu, mean=np.array([1.2, 3.3]), cov=np.diag((.5**2,.15**2))):
    return stats.multivariate_normal.pdf(logu, mean=mean , cov=cov)



def sample(logu, misfit, obs, observation_times, x0, TARGET_SIGMA=None, augment=True):
    
    #here u is in log scale
    #metropolis hasting
    
    if TARGET_SIGMA is None:
        TARGET_SIGMA = np.diag((.25,.25))
    u_init = logu
    u_prop = target_draw(u_init, TARGET_SIGMA)
    #################### for check the proposed candidate, comment this if no bug
    print(u_prop)
    prior_init = pdf_calc(u_init, np.array([1.2, 3.3]), np.diag((.5**2,.15**2)))
    prior_prop = pdf_calc(u_prop, np.array([1.2, 3.3]), np.diag((.5**2,.15**2)))
    
    like_init = misfit(logu, obs, observation_times, x0, augment=augment)
    like_prop = misfit(u_prop, obs, observation_times, x0, augment=augment)
    
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


    

def MH_trace(logu, obs, observation_times, x0, misfit, TARGET_SIGMA=None, CHAIN_LEN=5000, augment=True):
    # np.random.seed(0)
    if TARGET_SIGMA is None:
        #step size
        TARGET_SIGMA = np.diag((.25,.25))

    
    
    #beta, rho
    u_list = np.zeros((CHAIN_LEN,2))

    count=0
    for epoch in range(CHAIN_LEN):
        
        logu,acpt = sample(logu, misfit, obs, observation_times, x0, TARGET_SIGMA=TARGET_SIGMA, augment = augment)
        
        u_list[epoch] = logu
        count += acpt
    print('Acceptance ratio is: {}'.format(count/CHAIN_LEN))

    return u_list,count/CHAIN_LEN




    
if __name__ == '__main__':
    #np.random.seed()
    d = 3
    # if change obs from (x,y,z) to (x,y,z,x**2,y**2,z**2,xy,yz,xz) CES paper
    augment = False
    N = 10 
    x0 = -15 + 30 * np.random.random((N, d))   
    num = 360
    negini = 0
    t_1 = 0
    t_final = 4
    observation_times = np.linspace(t_1, t_final, num = num)
    #(x,y,z,x2,y2,z2,xy,yz,xz)
    _,obs1 = solve_lorenz(x0=x0, t=observation_times, sigma=10.0, beta=8./3, rho=28.0)
    obs = obs1[:,negini:,:].mean(axis=1) # #of trial N * 3
    if augment:
        obs2 = (obs1[:,negini:,:]**2).mean(axis=1)
        xy = (obs1[:,negini:,0]*obs1[:,negini:,1]).mean(axis=1, keepdims = True)
        yz = (obs1[:,negini:,1]*obs1[:,negini:,2]).mean(axis=1, keepdims = True)
        xz = (obs1[:,negini:,0]*obs1[:,negini:,2]).mean(axis=1, keepdims = True)
        obs = np.hstack((obs,obs2,xy,yz,xz))
    
    logu = gene_prior()
    #res = misfit(u, obs, observation_times[negini:], x0,augment=False)
    u_sample,apratio = MH_trace(logu, obs, observation_times[negini:], x0, misfit, 
                                TARGET_SIGMA= np.diag((.25,.25)), CHAIN_LEN=10, augment=False)
    pos_u = np.mean(np.exp(u_sample[1000:]),axis=0)
    _,G_u = solve_lorenz(x0=x0, t=observation_times, sigma=10.0, beta=pos_u[0], rho=pos_u[1])
    
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
        
    plot(obs)
    
    
    fig,axes = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=True,figsize=(16,4))
    for i, ax in enumerate(axes.flat):
        plt.axes(ax)
        plt.plot(np.exp(u_sample[:,0]),'o')
        plt.plot(np.exp(u_sample[:,1]),'o')
        #plt.title('Average $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
    
    