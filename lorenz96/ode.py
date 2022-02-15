#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The Lorenz96 system of differential equations
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)

__author__ = "Shuyi Li"
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

class lorenz96:
    """
    Lorenz96 ordinary differential equations
    dx_k/dt = - x_{k-1}*(x_{k-2} - x_{k+1}) - x_k + F - h*c*y_kbar
    1/c*dy_lk/dt = -b*y_{l+1,k}*(y_{l+12,k} - y_{l-1,k}) - y_lk + h/L*x_kbar
    """
    def __init__(self, x0=None, t=None, h=1, F=10, logc=np.log(10), b=10, L=10, K=36, **kwargs):
        """
        x0: initial state
        t: time points to solve the dynmics at
        (h, F, logc, b): parameters
        """
        self.L = L
        self.K = K
        self.n = (self.L+1)*self.K
        if x0 is None:
            self.num_traj = kwargs.get('num_traj',1)
            rng = np.random.RandomState(kwargs.get('randinit_seed')) if 'randinit_seed' in kwargs else np.random
            self.x0 = rng.random((self.num_traj, self.n))
        else:
            self.x0 = x0
            self.num_traj = x0.shape[0] if np.ndim(x0)>1 else 1
        if t is None:
            max_time = kwargs.get('max_time',10.)
            time_res = kwargs.get('time_res',200)
            self.t = np.linspace(0, max_time, time_res)
        else:
            self.t = t
        self.h = h
        self.F = F
        self.logc = logc
        self.b = b
        
        # count PDE solving times
        self.soln_count = np.zeros(4)
        # 0-3: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively
    
    
    def solveFwd(self, params=None, t=None):
        """
        Solve the forward equation
        """
        if params is None:
            params = (self.h, self.F, self.logc, self.b)
        elif type(params) is not tuple:
            params = tuple(params)
        if t is None:
            t = self.t
        sol = [integrate.odeint(self.lorenz_96_slow_fast, x0i, t, args=params) for x0i in self.x0]
        #xts:(time_res, K), yts:(time_res, K, L)
        xts, yts= [self.unflatten_time_series(soli ) for soli in sol][0]
                #cont_soln = [sol_i.sol for sol_i in sol]
        self.soln_count[0] += 1
        return xts,yts #cont_soln
    
    
    def solve(self, params=None, t=None, opt='fwd', **kwargs):
        """
        Solve lorenz96 dynamics
        """
        if params is None:
            params = (self.h, self.F, self.logc, self.b)
        if t is None:
            t = self.t
        if opt == 'fwd':
            out = self.solveFwd(params, t) # xts:(time_res, K), yts:(time_res, K, L)
        elif opt == 'adj':
            out = self.solveAdj(params, t, **kwargs)
        else:
            out = None
        return out
    
    
    def index(self, x, i):
        return np.roll(x, -i, axis=-1)
    
    def ghost_cells(self, x, pad):
        pads = [(0,0)]*(x.ndim - 1) + [pad]
        return np.pad(x, pads, 'wrap')
    
    
    def remove_ghost_cells(self, x, pad, axis=-1):
        n = x.shape[axis]
        a, b = pad
        inds = np.r_[a:n-b]
        return x.take(inds, axis=axis)
    
    
    def x_src(self, x, y, h=1, F=10, logc=np.log(10), L=10, **_):
        # x source terms
        pad = [2,2]
        x = self.ghost_cells(x, pad)
        yb = y.reshape((-1, L)).mean(axis=-1)
        yg = self.ghost_cells(yb, pad)
    
        dx = np.empty_like(x)
        for k in range(2, dx.shape[0]-1):
            dx[k] = -x[k-1] * (x[k-2] - x[k+1]) - x[k] + F - h*np.exp(logc)* yg[k]
        return self.remove_ghost_cells(dx, pad)
    
    def y_src(self, x, y, h=1, b=10, L=10, **_):
        pad = [2, 2]
    
        npad = 2
        y = np.pad(y, npad, mode='wrap')
        dy = np.empty_like(y)
        
        for j in range(1, dy.shape[0]-2):
            k = (j - npad) // L
            dy[j] = (-b * y[j+1] * (y[j+2] - y[j-1]) - y[j] + h / L * x[k])#np.exp(logc)*
        
        dy = dy[pad[0]:-pad[1]]
        return dy
    
    
    def flatten_state(self, x, y):
        return np.concatenate([x, y])
    
    
    def unflatten_state(self, x, K):
        x, y = x[:K], x[K:]
        return x, y
    
    def unflatten_time_series(self, x):
        x, y = x[:,:self.K], x[:,self.K:]
        y = y.reshape((-1, self.K, self.L))
        return x, y
    
    
    def lorenz_96_slow_fast(self, x, t, h, F, logc, b, **kwargs):
        
        x, y = self.unflatten_state(x, self.K)
        dx = self.x_src(x, y, h, F, logc, self.L, **kwargs)
        dy = self.y_src(x, y, h, b, self.L, **kwargs)
        return self.flatten_state(dx, dy)
        
    
    def test_ghost_cells(self,):
        #only for testing
        pad = [2, 1]
        x = np.arange(20).reshape((4, 5))
        y = self.ghost_cells(x, pad)
        assert y.shape == (4, 2 +1 + 5)
        
        np.testing.assert_array_equal(self.remove_ghost_cells(y, pad), x)
        
        
    def test_flatten_unflatten(self,):
        #only for testing
        K, L = 36, 10
        x = np.random.rand(36)
        y = np.random.rand(36*10)
        
        ans = self.unflatten_state(self.flatten_state(x, y), K)
        np.testing.assert_array_equal(x, ans[0])
        np.testing.assert_array_equal(y, ans[1])
        
    
    def plot_soln(self, x_t=None, **kwargs):
        """
        Plot the solution
        need to be modified
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
    from mpl_toolkits.mplot3d import Axes3D
    #### -- demonstration -- ####
    num_traj = 10
    t = np.linspace(0, 10, 100)
    L, K = 10, 2
    n = (L+1) * K
    x_t = np.zeros((num_traj,100,2*K))
    for i in range(num_traj):
        x0 = np.random.randn(num_traj,n)
        
        ode = lorenz96(x0=x0, t=t, L=L, K=K)
        xts, yts = ode.solve()
        yts=yts.mean(axis=2)
        x_t[i] = np.hstack((xts,yts))
        
    
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    #ode.plot_soln(ax=ax1,angle=10)
    for i in range(min(K,3)):
        if i==0:
            ax2_i = fig.add_subplot(3,2,(i+1)*2)
        else:
            ax2_i = fig.add_subplot(3,2,(i+1)*2, sharex=ax2_i)
        ax2_i.plot(ode.t, x_t[:,:,i+K].T) #i+K for first 3 y_k, x_t[:,:,i] for first 3 x_k
        # ax2_i.set_title('Trajectories of $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
        ax2_i.set_ylabel({0:'x',1:'y',2:'z'}[i], rotation='horizontal')
        if i==2:
            ax2_i.set_xlabel('t')
    plt.savefig(os.path.join(os.getcwd(),'properties/multi_traj.png'),bbox_inches='tight')
                             
    '''
    
   
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
    '''