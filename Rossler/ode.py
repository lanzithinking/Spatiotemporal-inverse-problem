#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The Rossler system of differential equations
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Mirjeta Pasha"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

'''
# import modules
import numpy as np
from scipy import integrate, interpolate
import matplotlib.pyplot as plt

class rossler:
    """
    Rossler ordinary differential equations
    dx_1/dt = - x_2 - x_3
    dx_2/dt = x_1 + a x_2
    dx_3/dt = b + (x_1 - c) x_3
    """
    def __init__(self, x0=None, t=None, a=0.2, b=0.2, c=5.7, **kwargs):
        """
        x0: initial state
        t: time points to solve the dynmics at
        (a, b, c): parameters
        """
        if x0 is None:
            self.num_traj = kwargs.get('num_traj',1)
            rng = np.random.RandomState(kwargs.get('randinit_seed')) if 'randinit_seed' in kwargs else np.ranodm
            self.x0 = -7 + 14 * rng.random((self.num_traj, 3))
        else:
            self.x0 = x0
            self.num_traj = x0.shape[0] if np.ndim(x0)>1 else 1
        if t is None:
            max_time = kwargs.get('max_time',4.)
            time_res = kwargs.get('time_res',1000)
            self.t = np.linspace(0, max_time, time_res)
        else:
            self.t = t
        self.a = a
        self.b = b
        self.c = c
        
        # count PDE solving times
        self.soln_count = np.zeros(4)
        # 0-3: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively
    
    # def _dx(self, x, t, a, b, c):
    def _dx(self, t, x, a, b, c):
        """
        Time derivative of Rossler dynamics
        """
        return [- x[1] - x[2], x[0] + a * x[1], b + (x[0]-c) * x[2]]
    
    def solveFwd(self, params=None, t=None):
        """
        Solve the forward equation
        """
        if params is None:
            params = (self.a, self.b, self.c)
        elif type(params) is not tuple:
            params = tuple(params)
        if t is None:
            t = self.t
        # x_t = np.asarray([integrate.odeint(self._dx, x0i, t, args=params, tfirst=True) for x0i in self.x0]) # (num_traj, time_res, 3)
        sol = [integrate.solve_ivp(self._dx, (min(t),max(t)), x0i, t_eval=t, args=params, dense_output=True,) for x0i in self.x0]
        x_t = np.asarray([sol_i.y.T for sol_i in sol]) # (num_traj, time_res, 3)
        cont_soln = [sol_i.sol for sol_i in sol]
        self.soln_count[0] += 1
        return x_t, cont_soln
    
    # def _dlmd(self, lmd, t, a, b, c, x_f, g_f):
    def _dlmd(self, t, lmd, a, b, c, x_f, g_f):
        """
        Time derivative of adjoint equation
        """
        x = x_f(t)
        g = g_f(t)
        return [ lmd[1] + lmd[2] - g[0], 
                -lmd[0] - a * lmd[1] - g[1], 
                -lmd[0]*x[2] - lmd[2]*(x[0] - c) - g[2]]
    
    def solveAdj(self, params=None, t=None, sol=None, msft=None, **kwargs):
        """
        Solve the adjoint equation
        """
        if params is None:
            params = (self.a, self.b, self.c)
        elif type(params) is not tuple:
            params = tuple(params)
        if t is None:
            t = self.t
        if sol is None:
            sol = self.solveFwd(params, t)
        if msft is None:
            from misfit import misfit
            msft = misfit(self, t)
        g = msft.grad(sol) # (num_traj, 3 or 9) for avg_traj; (num_traj, time_res, 3) for non-avg_traj
        cont_soln = kwargs.get('cont_soln')
        lmd_t = []
        for i in range(sol.shape[0]):
            xf_i = cont_soln[i] if cont_soln is not None else interpolate.interp1d(t, sol[i], axis=0, fill_value='extrapolate')
            if msft.STlik or np.ndim(g)==3:
                gf_i = lambda t_: interpolate.interp1d(t, g[i], axis=0, fill_value='extrapolate')(t_)/(len(t) if msft.avg_traj else 1)
            else:
                def gf_i(t_):
                    g_i = g[i,:3]
                    if msft.avg_traj=='aug':
                        x_it = xf_i(t_)
                        g_i += g[i,3:6]*x_it*2
                        g_i += (g[i,6:]*x_it[[1,0,2]])[[0,2,1]] + (g[i,6:]*x_it[[0,2,1]])[[1,0,2]]
                    return g_i/len(t)
            lmd_t.append(integrate.odeint(self._dlmd, np.zeros(3), t, args=params+(xf_i, gf_i), tfirst=True)[::-1])
        lmd_t = np.array(lmd_t) # (num_traj, time_res, 3)
        self.soln_count[1] += 1
        return lmd_t
    
    def solve(self, params=None, t=None, opt='fwd', **kwargs):
        """
        Solve Rossler dynamics
        """
        if params is None:
            params = (self.a, self.b, self.c)
        if t is None:
            t = self.t
        if opt == 'fwd':
            out = self.solveFwd(params, t)[0] # (num_traj, time_res, 3)
        elif opt == 'adj':
            out = self.solveAdj(params, t, **kwargs)
        else:
            out = None
        return out
    
    def plot_soln(self, x_t=None, **kwargs):
        """
        Plot the solution
        """
        if x_t is None:
            x_t = self.solve()
        num_traj = x_t.shape[0]
        
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        # ax.axis('off')
        
        # prepare the axes limits
        ax.set_xlim((-15, 15))
        ax.set_ylim((-15, 15))
        ax.set_zlim((-5, 15))
        
        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, num_traj))
        
        for i in range(num_traj):
            x1, x2, x3 = x_t[i,:,:].T
            lines = ax.plot(x1, x2, x3, '-', c=colors[i])
            plt.setp(lines, linewidth=2)
        
        angle = kwargs.pop('angle',0.0)
        ax.view_init(30, angle)
        # plt.show()
        return ax

if __name__ == '__main__':
    np.random.seed(2021)
    import os
    import pandas as pd
    import seaborn as sns
    
    #### -- demonstration -- ####
    num_traj = 10
    ode = rossler(num_traj=num_traj,max_time=5,time_res=1000)
    x_t = ode.solve()
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ode.plot_soln(ax=ax1,angle=10)
    for i in range(3):
        if i==0:
            ax2_i = fig.add_subplot(3,2,(i+1)*2)
        else:
            ax2_i = fig.add_subplot(3,2,(i+1)*2, sharex=ax2_i)
        ax2_i.plot(ode.t, x_t[:,:,i].T)
        # ax2_i.set_title('Trajectories of $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
        ax2_i.set_ylabel({0:'x',1:'y',2:'z'}[i], rotation='horizontal')
        if i==2:
            ax2_i.set_xlabel('t')
    plt.savefig(os.path.join(os.getcwd(),'properties/multi_traj.png'),bbox_inches='tight')
    
    #### -- multiple short trajectories -- ####
    # define the Rossler ODE
    num_traj = 5000
    ode_multrj = rossler(num_traj=num_traj,max_time=5,time_res=10000)
    # generate n trajectories
    xt_multrj = ode_multrj.solve()
    # ode_multrj.plot_soln()
    # plt.show()
    # average trajectory
    xt_multrj_avg = xt_multrj.mean(axis=1)
    print("The mean of time-averages for {:d} short trajectories is: ".format(num_traj))
    print(["{:.4f}," .format(i) for i in xt_multrj_avg.mean(0)])
    # plot
    # fig,axes = plt.subplots(nrows=1,ncols=3,sharex=False,sharey=True,figsize=(16,4))
    # for i, ax in enumerate(axes.flat):
    #     plt.axes(ax)
    #     plt.hist(xt_multrj_avg[:,i])
    #     plt.title('Average $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
    # plt.show()
    xts_avg = pd.DataFrame(xt_multrj_avg,columns=['x_avg','y_avg','z_avg'])
    g = sns.PairGrid(xts_avg, diag_sharey=False, size=3)
    g.map_upper(sns.scatterplot, size=5)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    g.savefig(os.path.join(os.getcwd(),'properties/multi_traj_avg.png'),bbox_inches='tight')
    
    #### -- one long trajectories -- ####
    # define the Rossler ODE
    num_traj = 1
    ode_multrj = rossler(num_traj=num_traj,max_time=5000,time_res=50000)
    # generate n trajectories
    xt_multrj = ode_multrj.solve()
    # ode_multrj.plot_soln()
    # plt.show()
    # average trajectory
    xt_multrj_avg = xt_multrj.mean(axis=1)
    print("The mean of time-averages for {:d} long trajectories is: ".format(num_traj))
    print(["{:.4f}," .format(i) for i in xt_multrj_avg.mean(0)])
    # plot
    # pcnt=.1
    # fig,axes = plt.subplots(nrows=3,ncols=2,sharex=False,sharey=False,figsize=(16,10))
    # for k, ax in enumerate(axes.flat):
    #     plt.axes(ax)
    #     i=k//2; j=k%2
    #     if j==0:
    #         plt.plot(ode_multrj.t[:np.floor(len(ode_multrj.t)*pcnt).astype(int)], xt_multrj[:,:np.floor(len(ode_multrj.t)*pcnt).astype(int),i].T)
    #         plt.title('First '+str(pcnt*100)+'% Trajectories of $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
    #     elif j==1:
    #         plt.plot(ode_multrj.t[-np.floor(len(ode_multrj.t)*pcnt).astype(int):], xt_multrj[:,-np.floor(len(ode_multrj.t)*pcnt).astype(int):,i].T)
    #         plt.title('Last '+str(pcnt*100)+'% Trajectories of $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
    # plt.show()
    xt = pd.DataFrame(xt_multrj.squeeze()[::10],columns=['x','y','z'])
    g = sns.PairGrid(xt, diag_sharey=False, size=3)
    g.map_upper(sns.scatterplot, size=5)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot)
    g.savefig(os.path.join(os.getcwd(),'properties/long_traj.png'),bbox_inches='tight')