#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:29:58 2022

@author: apple
"""
import argparse
from Rossler import *
sys.path.append( "../" )
# from util import *
from optimizer.EnK import EnK



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=1)
    #parser.add_argument('-STlik', action="store_true"), specify -STlik means True otherwise False
    parser.add_argument('STlik', nargs='?', type=bool, default=False)
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    args = parser.parse_args()

    
    np.random.seed(2021)
    # define Bayesian inverse problem
    num_traj = 1
    ode_params = {'a':0.2, 'b':0.2, 'c':5.7}
    t_init = 1000
    t_final = 1100
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = 'aug' if not args.STlik else False#False
    var_out = True
    STlik = args.STlik#True
    ode_init = -7 + 14 * np.random.random((num_traj, 3))
    rsl = Rossler(num_traj=num_traj, ode_params=ode_params, ode_init=ode_init, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=2021, STlik=STlik)
    
    
    def G(u):
        #rsl.ode.solveFwd(u[0])[0]
        G_u = np.asarray( [rsl.misfit.observe(sol=rsl.ode.solve(params=u_i,t=rsl.obs_times),var_out=False) for u_i in np.exp(u)])
        
        return G_u.reshape((len(u),-1)) if rsl.misfit.STlik else G_u.squeeze()
    
    #STlik,  1*#time*3 -> 1*3#time  
    y = (rsl.misfit.obs).reshape((1,-1)) if rsl.misfit.STlik else rsl.misfit.obs        
    data={'obs':y,'size':y.size,'cov': rsl.misfit.stgp.tomat() if rsl.misfit.STlik else np.diag(rsl.misfit.nzvar[0])}
    
     
    def runek(max_iter=10, J=500, stp_sz=[1,.1], nz_lvl=1, err_thld=1e-1, alg='EKS', seed=2021, save='True'):
        ''' EKS/EKI 
        max_iter: iteration
        J: ensemble size
        stp_sz: step size for discretized Kalman dynamics(EKI,EKS)
        nz_lvl: level of perturbed noise to data, 0 means no perturbation
        err_thld: threshold factor for stopping criterion
        reg: indicator of whether to implement regularization in optimization (EKI)
        adpt: indicator of whether to implement adaptation in stepsize (EKS)
        '''
        
        np.random.seed(seed)
        
        # initial ensemble
        pri_samp=lambda n=1: (rsl.prior.sample(n))
        u=pri_samp(J)
        prior={'mean':rsl.prior.mean,'cov':np.diag(np.square(rsl.prior.std)),'sample':pri_samp}
        
        enk=EnK(u,G,data,prior,stp_sz=stp_sz[alg=='EKS'],nz_lvl=nz_lvl,err_thld=err_thld,alg=alg,adpt=True)
        # run ensemble Kalman algorithm
        if save:
            ek_fun=enk.run
            ek_args=(max_iter,True)
            savepath,filename=ek_fun(*ek_args)
            
            # append ODE information including the count of solving
            filename_=os.path.join(savepath,filename+'.pckl')
            filename=os.path.join(savepath,'rossler_'+'ST_'+str(rsl.misfit.STlik)[0]+'_'+filename+'.pckl') # change filename
            os.rename(filename_, filename)
            f=open(filename,'ab')
            
            pickle.dump([ode_params,STlik,avg_traj,y],f)
            f.close()
        else:            
            max_iter=max_iter
            return (enk.run(max_iter=max_iter) )
        
    def run(repeat=10, stp_sz=[1,.1], nz_lvl=1, err_thld=1e-1, alg='EKS'):
        # try multiple experiments with same setting -> std
        
        for j in [100,200,500,1000]: 
            seed=2021
            for i in range(repeat):
                runek(int(5000/j), j, stp_sz, nz_lvl, err_thld, alg, seed)
                seed+=1
                
    run(alg=args.algs[args.algNO])
                
if __name__ == '__main__':
    
    main()
        
    
            
        
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
