#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Get relative error of mean for uncertainty field u in Advection-Diffusion inverse problem.
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Modified for STIP August 2021 @ ASU
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp





seed=2020

true_param = {'a':0.2, 'b':0.2, 'c':5.7}
true_param = list(true_param.values())



# algorithms
alg_names=('simEKI','simEKS','STlikEKI','STlikEKS')
algs=('ST_F_EKI','ST_F_EKS','ST_T_EKI','ST_T_EKS')
num_algs=len(algs)
# store results
lik_mdls=('J=100','J=200','J=500','J=1000')
num_mdls=len(lik_mdls)
rem_m=np.zeros((num_mdls,num_algs))
rem_s=np.zeros((num_mdls,num_algs))
# obtain estimates
folder = './result'
pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]




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
                
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f)
                extra_info=pickle.load(f)                
                cur = [np.linalg.norm(f_read[0][j,:] - true_param )/np.linalg.norm(true_param) for j in range(int(5000/int(lik_mdls[m][2:]) ))]
                errs.append(np.min(cur))
                num_read+=1
                f.close()
                print(f_i+' has been read!')
        if num_read>0:             
            errs = np.stack(errs)
            rem_m[m,i] = np.median(errs)
            rem_s[m,i] = errs.std()
                
        print('%d experiment(s) have been processed for %s algorithm with %s model.' % (num_read, algs[i], lik_mdls[m]))
       
# save
import pandas as pd
rem_m = pd.DataFrame(data=rem_m,index=lik_mdls,columns=alg_names[:num_algs])
rem_s = pd.DataFrame(data=rem_s,index=lik_mdls,columns=alg_names[:num_algs])
rem_m.to_csv(os.path.join(folder,'REM-mean.csv'),columns=alg_names[:num_algs])
rem_s.to_csv(os.path.join(folder,'REM-std.csv'),columns=alg_names[:num_algs])