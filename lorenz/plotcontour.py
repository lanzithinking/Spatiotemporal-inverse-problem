#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 18:19:04 2021

@author: apple
"""
from lorenz_shuyi import *

def contour_misfit(h1st,h1e,h2st,h2e, lorenz, drop=1, num=1000, plot=True):
    # drop: which hyperparameter to drop, sigma=1, beta=2, rho=3
    h1grid = np.linspace(h1st, h1e, num=num)
    h2grid = np.linspace(h2st, h2e, num=num)
    misfit = np.zeros((num,num))
                                                                                   
    for i in range(num):
        for j in range(num):
            
            misfit[i,j] = lorenz.misfit([np.log(h1grid[i]), np.log(h2grid[j])], drop=drop)
            
    if plot:
        fig, axes = plt.subplots(1, 1, figsize=(4,4),sharey=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        name = ['sigma','beta','rho']
        name.pop(name.index(name[drop-1]))
        axes.contourf(h1grid, h2grid, misfit)
        axes.set_xlabel(name[0])
        axes.set_ylabel(name[1])
        axes.set_title('STlik:{0}'.format(str(lorenz.STlik))) 
        plt.show()
        #
        #axes[0].set_xlabel('RESTAURANT_CATEGORY',fontweight='bold')

    return misfit 


if __name__ == '__main__':
    np.random.seed(2021)
    d = 3
    # if change obs from (x,y,z) to (x,y,z,x**2,y**2,z**2,xy,yz,xz) CES paper 9*9
    # (g(u)-obs)cov^(-1)(g(u)-obs) 10*9
    augment = True
    N = 10 #1 for STLik 10 for not
    x0 = -15 + 30 * np.random.random((N, d))   
    
    time_resolution = 40
    negini = 0
    t_1 = 0
    t_final = 20
    observation_times = np.linspace(t_1, t_final, num = time_resolution*t_final+1)
    
    #construct lorenz problem
    lorenz = Lorenz(observation_times[negini:], x0, obs=None, augment = augment, STlik=False)
    
    misfit23 = contour_misfit(h1st=2.4,h1e=2.85, h2st=27.4, h2e=28.5, lorenz, drop=1, num=100, plot=True)
    misfit13 = contour_misfit(9.4, 10.5, 27.4, 28.5, lorenz, drop=2, num=20, plot=True)
    misfit12 = contour_misfit(9.4, 10.5, 2.4, 2.85, lorenz, drop=3, num=20, plot=True)
    
    augment = False
    N = 1 #1 for STLik 10 for not
    x0 = -15 + 30 * np.random.random((N, d))   
    lorenz = Lorenz(observation_times[negini:], x0, obs=None, augment = augment, STlik=True)
    
    misfit23lik = contour_misfit(h1st=2.4,h1e=2.85, h2st=27.4, h2e=28.5, lorenz=lorenz, drop=1, num=20, plot=True)
    misfit13lik = contour_misfit(9.4, 10.5, 27.4, 28.5, lorenz, drop=2, num=20, plot=True)
    misfit12lik = contour_misfit(9.4, 10.5, 2.4, 2.85, lorenz, drop=3, num=20, plot=True)
