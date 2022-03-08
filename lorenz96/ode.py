#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
The Lorenz96 system of differential equations
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
from scipy import integrate, interpolate
import matplotlib.pyplot as plt

class lrz96:
    """
    Lorenz96 ordinary differential equations
    dx_k/dt = - x_{k-1}*(x_{k-2} - x_{k+1}) - x_k + F - h*c*\bar{y}_k
    1/c*dy_lk/dt = -b*y_{l+1,k}*(y_{l+2,k} - y_{l-1,k}) - y_lk + h/L*x_k
    """
    def __init__(self, x0=None, t=None, K=36, L=10, h=1, F=10, logc=np.log(10), b=10, **kwargs):
        """
        x0: initial state
        t: time points to solve the dynmics at
        (h, F, logc, b): parameters
        """
        self.K = K
        self.L = L
        self.dim = self.K*(1+self.L)
        if x0 is None:
            self.num_traj = kwargs.get('num_traj',1)
            rng = np.random.RandomState(kwargs.get('randinit_seed')) if 'randinit_seed' in kwargs else np.random
            self.x0 = -1 +  2*rng.random((self.num_traj, self.dim))
            self.x0[:,:self.K] *= 10
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
        
        # count ODE solving times
        self.soln_count = np.zeros(4)
        # 0-3: number of solving (forward,adjoint,2ndforward,2ndadjoint) equations respectively
    
    # def _dx(self, x, t, h, F, logc, b):
    def _dx(self, t, x, h, F, logc, b):
        """
        Time derivative of Lorenz96 dynamics
        """
        c = np.exp(logc)
        X = x[:self.K]; Y = x[self.K:].reshape((self.L, self.K))
        dX = -np.roll(X,1)*(np.roll(X,2)-np.roll(X,-1)) -X + F - h*c*Y.mean(axis=0)
        Y_flatF = Y.flatten('F')
        dY = c*( -b*np.reshape(np.roll(Y_flatF,-1)*(np.roll(Y_flatF,-2)-np.roll(Y_flatF,1)),(self.L,self.K),'F') -Y + h/self.L * X[None,:] ) # (L,K)
        return np.append(dX,dY.flatten())
    
    def solveFwd(self, params=None, t=None, solver='solve_ivp'):
        """
        Solve the forward equation
        """
        if params is None:
            params = (self.sigma, self.beta, self.rho)
        elif type(params) is not tuple:
            params = tuple(params)
        if t is None:
            t = self.t
        if solver=='odeint':
            x_t = np.asarray([integrate.odeint(self._dx, x0i, t, args=params, tfirst=True) for x0i in self.x0]) # (num_traj, time_res, K(1+L))
        elif solver=='solve_ivp':
            sol = [integrate.solve_ivp(self._dx, (min(t),max(t)), x0i, t_eval=t, args=params, dense_output=False,) for x0i in self.x0]
            x_t = np.asarray([sol_i.y.T for sol_i in sol]) # (num_traj, time_res, K(1+L))
            # cont_soln = [sol_i.sol for sol_i in sol]
        else:
            raise ValueError('Solver not recognized.')
        self.soln_count[0] += 1
        return x_t, #cont_soln
    
    # def _dlmd(self, lmd, t, h, F, logc, b, x_f, g_f):
    def _dlmd(self, t, lmd, h, F, logc, b, x_f, g_f):
        """
        Time derivative of adjoint equation
        """
        raise NotImplementedError('The derivative of adjoint equation is not implemented.')
    
    def solveAdj(self, params=None, t=None, sol=None, msft=None, **kwargs):
        """
        Solve the adjoint equation
        """
        raise NotImplementedError('The adjoint solver is not implemented.')
    
    def solve(self, params=None, t=None, opt='fwd', **kwargs):
        """
        Solve lorenz96 dynamics
        """
        if params is None:
            params = (self.h, self.F, self.logc, self.b)
        if t is None:
            t = self.t
        if opt == 'fwd':
            out = self.solveFwd(params, t, **kwargs)[0] # (num_traj, time_res, K(1+L))
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
        X_t = x_t[:,:,:self.K]
        Y_t = x_t[:,:,self.K:].reshape((self.num_traj,-1,self.L,self.K))
        
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        # ax.axis('off')
        
        # prepare the axes limits
        ax.set_xlim((0, 10)); ax.set_xlabel('$X_1$')
        ax.set_ylim((-1, 1)); ax.set_ylabel('$Y_{1,1}$')
        ax.set_zlim((-1, 1)); ax.set_zlabel('$Y_{2,1}$')
        
        # choose a different color for each trajectory
        colors = plt.cm.jet(np.linspace(0, 1, num_traj))
        
        for i in range(num_traj):
            x1, x2, x3 = X_t[i,:,0], Y_t[i,:,0,0], Y_t[i,:,1,0]
            lines = ax.plot(x1, x2, x3, '-', c=colors[i])
            plt.setp(lines, linewidth=2)
        
        angle = kwargs.pop('angle',0.0)
        ax.view_init(30, angle)
        # plt.show()
        return ax

if __name__ == '__main__':
    np.random.seed(2021)
    # import os
    # import pandas as pd
    # import seaborn as sns
    # sns.set(font_scale=1.1)
    #
    # #### -- demonstration -- ####
    # #### -- one short trajectory -- ####
    # num_traj = 1
    # K, L = 2, 10
    # ode = lrz96(num_traj=num_traj,K=K,L=L,max_time=3,time_res=1000)
    # x_t = ode.solve().squeeze()
    # X = x_t[:,:ode.K]; Y = x_t[:,ode.K:].reshape((-1,ode.L,ode.K))
    # fig = plt.figure(figsize=(18,6))
    # ax1 = fig.add_subplot(1,3,1, projection='3d')
    # ode.plot_soln(ax=ax1,angle=10)
    # for i in range(4):
    #     if i//2==0:
    #         ax2_i = fig.add_subplot(2,3,(i//2)*3+2+i%2)
    #     else:
    #         ax2_i = fig.add_subplot(2,3,(i//2)*3+2+i%2, sharex=ax2_i)
    #     if i%2==0:
    #         ax2_i.plot(ode.t, X[:,i//2])
    #         ax2_i.set_ylabel('$X_{}$'.format(i//2+1), rotation='horizontal')
    #     else:
    #         ax2_i.plot(ode.t, Y[:,:,i//2].mean(axis=1))
    #         ax2_i.plot(ode.t, Y[:,:,i//2], color='grey',alpha=.4)
    #         ax2_i.set_ylabel('$Y_{\cdot,%d}$' % (i//2+1), rotation='horizontal')
    #     if i//2==1:
    #         ax2_i.set_xlabel('t')
    # plt.savefig(os.path.join(os.getcwd(),'properties/single_traj.png'),bbox_inches='tight')
    #
    # #### -- multiple short trajectories -- ####
    # num_traj = 3
    # K, L = 2, 10
    # ode = lrz96(num_traj=num_traj,K=K,L=L,max_time=3,time_res=1000)
    # x_t = ode.solve()
    # fig = plt.figure(figsize=(12,6))
    # ax1 = fig.add_subplot(1,2,1, projection='3d')
    # ode.plot_soln(ax=ax1,angle=10)
    # for i in range(3):
    #     if i==0:
    #         ax2_i = fig.add_subplot(3,2,(i+1)*2)
    #     else:
    #         ax2_i = fig.add_subplot(3,2,(i+1)*2, sharex=ax2_i)
    #     ax2_i.plot(ode.t, x_t[:,:,i*ode.K].T)
    #     # ax2_i.set_title('Trajectories of '+('$X_1$','$Y_{1,1}$','$Y_{2,1}$')[i]+'$(t)$')
    #     ax2_i.set_ylabel(('$X_1$','$Y_{1,1}$','$Y_{2,1}$')[i], rotation='horizontal')
    #     if i==2:
    #         ax2_i.set_xlabel('t')
    # plt.savefig(os.path.join(os.getcwd(),'properties/multi_traj.png'),bbox_inches='tight')
    #
    # #### -- multiple short trajectories -- ####
    # # define the Lorenz96 ODE
    # num_traj = 500
    # ode_multrj = lrz96(num_traj=num_traj,K=K,L=L,max_time=5,time_res=1000)
    # # generate n trajectories
    # xt_multrj = ode_multrj.solve()
    # # ode_multrj.plot_soln()
    # # plt.show()
    # # average trajectory
    # xt_multrj_avg = xt_multrj.mean(axis=1)
    # print("The mean of time-averages for {:d} short trajectories is: ".format(num_traj))
    # print(["{:.4f}," .format(i) for i in xt_multrj_avg.mean(0)])
    # # plot
    # # fig,axes = plt.subplots(nrows=1,ncols=3,sharex=False,sharey=True,figsize=(16,4))
    # # for i, ax in enumerate(axes.flat):
    # #     plt.axes(ax)
    # #     plt.hist(xt_multrj_avg[:,i*self.L])
    # #     plt.title('Average '+('$X_1$','$Y_{1,1}$','$Y_{2,1}$')[i]+'$(t)$')
    # # plt.show()
    # xts_avg = pd.DataFrame(xt_multrj_avg[:,np.arange(3)*ode.K],columns=['x1_avg','y11_avg','y21_avg'])
    # g = sns.PairGrid(xts_avg, diag_sharey=False, size=3)
    # g.map_upper(sns.scatterplot, size=5)
    # g.map_lower(sns.kdeplot)
    # g.map_diag(sns.kdeplot)
    # g.savefig(os.path.join(os.getcwd(),'properties/multi_traj_avg.png'),bbox_inches='tight')
    #
    # #### -- one long trajectories -- ####
    # # define the Lorenz96 ODE
    # num_traj = 1
    # ode_multrj = lrz96(num_traj=num_traj,K=K,L=L,max_time=1000,time_res=10000)
    # # generate n trajectories
    # xt_multrj = ode_multrj.solve()
    # # ode_multrj.plot_soln()
    # # plt.show()
    # # average trajectory
    # xt_multrj_avg = xt_multrj.mean(axis=1)
    # print("The mean of time-averages for {:d} long trajectories is: ".format(num_traj))
    # print(["{:.4f}," .format(i) for i in xt_multrj_avg.mean(0)])
    # # plot
    # # pcnt=.1
    # # fig,axes = plt.subplots(nrows=3,ncols=2,sharex=False,sharey=False,figsize=(16,10))
    # # for k, ax in enumerate(axes.flat):
    # #     plt.axes(ax)
    # #     i=k//2; j=k%2
    # #     if j==0:
    # #         plt.plot(ode_multrj.t[:np.floor(len(ode_multrj.t)*pcnt).astype(int)], xt_multrj[:,:np.floor(len(ode_multrj.t)*pcnt).astype(int),i].T)
    # #         plt.title('First '+str(pcnt*100)+'% Trajectories of $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
    # #     elif j==1:
    # #         plt.plot(ode_multrj.t[-np.floor(len(ode_multrj.t)*pcnt).astype(int):], xt_multrj[:,-np.floor(len(ode_multrj.t)*pcnt).astype(int):,i].T)
    # #         plt.title('Last '+str(pcnt*100)+'% Trajectories of $'+{0:'x',1:'y',2:'z'}[i]+'(t)$')
    # # plt.show()
    # xt = pd.DataFrame(xt_multrj.squeeze()[::10,np.arange(3)*ode.K],columns=['$X_1$','$Y_{1,1}$','$Y_{2,1}$'])
    # g = sns.PairGrid(xt, diag_sharey=False, size=3)
    # g.map_upper(sns.scatterplot, size=5)
    # g.map_lower(sns.kdeplot)
    # g.map_diag(sns.kdeplot)
    # for ax in g.axes.flatten():
    #     # rotate x axis labels
    #     # ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    #     # rotate y axis labels
    #     ax.set_ylabel(ax.get_ylabel(), rotation = 0)
    #     # set y labels alignment
    #     ax.yaxis.get_label().set_horizontalalignment('right')
    # g.savefig(os.path.join(os.getcwd(),'properties/long_traj.png'),bbox_inches='tight')
    #
    # #### -- steady state -- ####
    # import time
    # start = time.time()
    # ode = lrz96(max_time=1e4,time_res=10000)
    # steady_state = ode.solve()[:,-1,:]
    # end = time.time()
    # print('Time used is %.4f' % (end-start))
    # import pickle
    # f=open(os.path.join(os.getcwd(),'steady_state.pckl'),'wb')
    # pickle.dump(steady_state,f)
    # f.close()
    
    #### -- test solvers -- ####
    import time
    # ode = lrz96(max_time=10,time_res=100)
    ode = lrz96(t=np.linspace(100,110,100))
    solver = 'solve_ivp'
    start = time.time()
    sol1 = ode.solve(solver=solver)
    end = time.time()
    print('Time used by '+ solver+' is %.4f' % (end-start))
    solver = 'odeint'
    start = time.time()
    sol2 = ode.solve(solver=solver)
    end = time.time()
    print('Time used by '+ solver+' is %.4f' % (end-start))