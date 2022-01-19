#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:29:58 2022

@author: apple
"""

from Rossler import *
sys.path.append( "../" )
# from util import *
from optimizer.EnK import EnK

if __name__ == '__main__':

    seed=2021
    np.random.seed(seed)
    # define Bayesian inverse problem
    num_traj = 1
    ode_params = {'a':0.2, 'b':0.2, 'c':5.7}
    t_init = 1000
    t_final = 1100
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = 'aug'#False#
    var_out = True
    STlik = False#True
    rsl = Rossler(num_traj=num_traj, ode_params=ode_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik)
    
    def G(u):
        #rsl.ode.solveFwd(u[0])[0]
        G_u = np.asarray( [rsl.misfit.observe(sol=rsl.ode.solve(params=u_i,t=rsl.obs_times),var_out=False) for u_i in u])
        
        return G_u.reshape((len(u),-1)) if rsl.misfit.STlik else G_u.squeeze()
    
    #STlik,  1*#time*3 -> 1*3#time  
    y = (rsl.misfit.obs).reshape((1,-1)) if rsl.misfit.STlik else rsl.misfit.obs        
    data={'obs':y,'size':y.size,'cov': rsl.misfit.stgp.tomat() if rsl.misfit.STlik else np.diag(rsl.misfit.nzvar[0])}
    
     
    def runek(max_iter=10, J=500, stp_sz=[1,.1], nz_lvl=1, err_thld=1e-1, alg='EKS'):
        ''' EKS/EKI 
        max_iter: iteration
        J: ensemble size
        stp_sz: step size for discretized Kalman dynamics(EKI,EKS)
        nz_lvl: level of perturbed noise to data, 0 means no perturbation
        err_thld: threshold factor for stopping criterion
        reg: indicator of whether to implement regularization in optimization (EKI)
        adpt: indicator of whether to implement adaptation in stepsize (EKS)
        '''
        # initial ensemble
        pri_samp=lambda n=1: np.exp(rsl.prior.sample(n))
        u=pri_samp(J)
        prior={'mean':rsl.prior.mean,'cov':np.diag(np.square(rsl.prior.std)),'sample':pri_samp}
        
        eks=EnK(u,G,data,prior,stp_sz=stp_sz[alg=='EKS'],nz_lvl=nz_lvl,err_thld=err_thld,alg=alg,reg=True,adpt=True)
        # run ensemble Kalman algorithm
        max_iter=max_iter
        res_eks=eks.run(max_iter=max_iter)
        
        return res_eks
    
    
    
    def run(repeat=10, max_iter=10, J=500, stp_sz=[1,.1], nz_lvl=1, err_thld=1e-1, alg='EKS'):
        res_ekssiml = []
        for i in range(repeat):
            res_ekssim = runek(max_iter, J, stp_sz, nz_lvl, err_thld, alg)
            res_ekssiml.append(res_ekssim)
        return res_ekssiml
    
    
    #create table for difference
    def diff(res_enkl=[res_ekssim], real_params=list(ode_params.values()) ):
        diff = [0]*len(res_enkl)
        for i,res_enk in enumerate(res_enkl):
            diff[i] = np.linalg.norm(np.mean(res_enk[0],0) - real_params )/np.linalg.norm(real_params)
        
        return diff
        
    
    def collect(alg='EKI'):
        # try different combinations with J and iter
        res_ekl = []
        for i in [100,200,250,500,1000]:        
            res_ekl.append(  run(1, int(5000/i), i, alg=alg)[0] )
        return diff(res_ekl)
    
    erroreki = collect()
    erroreks = collect('EKS')
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
