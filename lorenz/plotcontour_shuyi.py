#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 18:19:04 2021

@author: apple
"""
#from lorenz_shuyi import *
from misfit import *

def contour_misfit_shuyi(h1st,h1e,h2st,h2e, lorenz, drop=1, num=1000, plot=True):
    # based on shuyi's class
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

def contour_misfit(h1st,h1e,h2st,h2e, mft, drop=1, num=1000, plot=True):
    # drop: which hyperparameter to drop, sigma=1, beta=2, rho=3
    h1grid = np.linspace(h1st, h1e, num=num)
    h2grid = np.linspace(h2st, h2e, num=num)
    misfit = np.zeros((num,num))
                                                                                   
    for i in range(num):
        for j in range(num):
            if drop==1:
                sol = mft.ode.solve(params=(10.0, h1grid[i], h2grid[j]), t=mft.obs_times)
            elif drop==2:
                sol = mft.ode.solve(params=(h1grid[i], 8/3, h2grid[j]), t=mft.obs_times)
            else:
                sol = mft.ode.solve(params=(h1grid[i], h2grid[j], 28), t=mft.obs_times)
            misfit[i,j] = mft.cost(sol) 
            
    if plot:
        fig, axes = plt.subplots(1, 1, figsize=(4,4),sharey=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.2)
        name = ['sigma','beta','rho']
        name.pop(name.index(name[drop-1]))
        #important, misfit(i,j) corresponding to ith h1, jth h2, whereas contour(i,j) -> ith h2, jth h1
        cf=axes.contourf(h1grid, h2grid, misfit.T)
        axes.set_xlabel(name[0])
        axes.set_ylabel(name[1])
        axes.set_title('STlik:{0}'.format(str(mft.STlik))) 
        fig.colorbar(cf, ax=axes)
        plt.show()
        #
        #axes[0].set_xlabel('RESTAURANT_CATEGORY',fontweight='bold')

    return h1grid,h2grid,misfit 

def plotcontour(p1=[7,13], p2=[-1/3,17/3], p3=[25,31], row=2, col=3):
    #p1,p2,p3 refer to domain of sigma,beta and rho when calculating the misfit
    global misfit
    np.random.seed(2021)
    row = row
    col = col
    fig, axes = plt.subplots(row, col, figsize=(14,10))
    plt.subplots_adjust(wspace=0.5, hspace=0.25)
    h1stlist = [p2[0], p1[0], p1[0]]
    h1elist = [p2[1], p1[1], p1[1]]
    h2stlist = [p3[0], p3[0], p2[0]]
    h2elist = [p3[1], p3[1], p2[1]]
   
    for i in range(row):
        num_traj=1; 
        avg_traj='aug' if i==0 else False
        mft = misfit(num_traj=num_traj, avg_traj=avg_traj, save_obs=False, STlik=(i==1))
        for j in range(col):
            name = ['sigma','beta','rho']
            name.pop(name.index(name[j]))
            h1grid,h2grid,misfitv = contour_misfit(h1st=h1stlist[j], h1e=h1elist[j], h2st=h2stlist[j], 
                                                  h2e=h2elist[j], mft=mft, drop=j+1, num=20, plot=False)
            cf=axes[i,j].contourf(h1grid, h2grid, misfitv.T)
            axes[i,j].set_xlabel(name[0])
            axes[i,j].set_ylabel(name[1])
            axes[i,j].set_title('STlik:{0}'.format(str(mft.STlik))) 
            fig.colorbar(cf, ax=axes[i,j])
    plt.show()
        

        
if __name__ == '__main__':
    '''
    #shuyi's class
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
    '''
    plotcontour()  
    plotcontour(p1=[8,12], p2=[2.4,2.85], p3=[25,31])  
    
