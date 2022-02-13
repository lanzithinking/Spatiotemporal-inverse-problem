#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Get prediction in rossler inverse problem.
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Modified: Shuyi Li
"""
from Rossler import *
#import os,pickle
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)



np.random.seed(2021)



# define Bayesian inverse problem
num_traj = 1
ode_params = {'a':0.2, 'b':0.2, 'c':5.7}
true_param = list(ode_params.values())
t_init = 1000
t_final = 1100
time_res = 100
obs_times = np.linspace(t_init, t_final, time_res)
avg_traj = False#False
var_out = True
STlik = True
ode_init = -7 + 14 * np.random.random((num_traj, 3))
rsl = Rossler(num_traj=num_traj, ode_params=ode_params, ode_init=ode_init, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=2021, STlik=STlik)
true_trj = rsl.misfit.obs

# selective locations to aggregate difference in observations
#cond = np.logical_or([abs(x-.25)<.01 or abs(x-.6)<.01 for x in adif.misfit.targets[:,0]],[abs(y-.4)<.01 or abs(y-.85)<.01 for y in adif.misfit.targets[:,1]]) # or abs(y-.85)<.01
#slab_idx = np.where(cond)[0]
#cond = [t>=1 and t<=4 for t in adif_pred.misfit.observation_times]
#obs_idx = np.where(cond)[0]



# algorithms
alg_names=('STlikEKI','STlikEKS')
algs=('ST_T_EKI','ST_T_EKS')
num_algs=len(algs)
# store results
lik_mdls=('J=100','J=200','J=500','J=1000')
num_mdls=len(lik_mdls)
rem_m=np.zeros((num_mdls,num_algs))
rem_s=np.zeros((num_mdls,num_algs))
# obtain estimates
folder = './result'
pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]

tl = len(rsl.misfit.obs_times)
pred_m=[np.zeros((num_mdls,rsl.misfit.obs.shape[2],len(rsl.misfit.obs_times))),np.zeros((num_algs,rsl.misfit.obs.shape[2],len(rsl.misfit.obs_times)))]
pred_std=[np.zeros((num_mdls,rsl.misfit.obs.shape[2],len(rsl.misfit.obs_times))),np.zeros((num_algs,rsl.misfit.obs.shape[2],len(rsl.misfit.obs_times)))]
#err_m=[np.zeros((num_mdls,len(rsl.misfit.obs_times))),np.zeros((num_algs,len(rsl.misfit.obs_times)))]
#err_std=[np.zeros((num_mdls,len(rsl.misfit.obs_times))),np.zeros((num_algs,len(rsl.misfit.obs_times)))]
err_m=np.zeros((num_mdls,num_algs))
err_s=np.zeros((num_mdls,num_algs))

for m in range(num_mdls):
    print('Processing '+lik_mdls[m]+' likelihood model...\n')
    # preparation for estimates
    num_samp=5000
    prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
    for i in range(num_algs):
        print('Getting estimates for '+algs[i]+' algorithm...')
        num_read=0
        errs=[]
        for f_i in pckl_files:
            if '_'+algs[i]+'_ensbl'+lik_mdls[m][2:]+'_' in f_i:
                fwdout_mean=0; fwdout_std=0; fwderr_mean=0; fwderr_std=0
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f)
                extra_info=pickle.load(f)   
                _,err,pred=f_read[:3]
                pred = pred[err!=0]               
                #5000*100*3
                pred = pred.reshape((pred.shape[0]*pred.shape[1], tl, len(true_param) ))
                fwdout_mean = pred.mean(axis=0) #100*3
                fwdout_std = (pred**2).mean(axis=0)    
                err = np.linalg.norm(fwdout_mean-true_trj[0,:,:])/np.linalg.norm(true_trj[0,:,:]) 
                errs.append(err)
                num_read+=1
                f.close()
                print(f_i+' has been read!')
        if num_read>0: 
            
            errs = np.stack(errs)
            err_m[m,i] = np.median(errs)
            err_s[m,i] = errs.std()
                
        print('%d experiment(s) have been processed for %s algorithm with %s model.' % (num_read, algs[i], lik_mdls[m]))
       
# save
import pandas as pd
rem_m = pd.DataFrame(data=err_m,index=lik_mdls,columns=alg_names[:num_algs])
rem_s = pd.DataFrame(data=err_s,index=lik_mdls,columns=alg_names[:num_algs])
rem_m.to_csv(os.path.join(folder,'prediction-mean.csv'),columns=alg_names[:num_algs])
rem_s.to_csv(os.path.join(folder,'prediction-std.csv'),columns=alg_names[:num_algs])




