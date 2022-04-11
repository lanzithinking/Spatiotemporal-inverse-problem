"""
find the best EnK output with the lowest error
Shiwei Lan @ ASU, August 2022
"""

import numpy as np

import os,sys
sys.path.append( "../" )
import pickle

SAVE=True

# algorithms and settings
algs=['EKI','EKS']
num_algs=len(algs)
lik_mdls=('simple','STlik')
num_mdls=len(lik_mdls)
ensbl_sz=500
max_iter=50

# preparation for estimates
folder = './analysis'
for m in range(num_mdls):
    print('Processing '+lik_mdls[m]+' likelihood model...\n')
    fld_m = folder+'/'+lik_mdls[m]
    # prepare data
    pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
    for a in range(num_algs):
        print('Working on '+algs[a]+' algorithm...')
        found=False
        best_f=''; min_err=np.inf
        # ensembles and forward outputs
        for f_i in pckl_files:
            if '_'+algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
                try:
                    f=open(os.path.join(fld_m,f_i),'rb')
                    loaded=pickle.load(f)
                    err=loaded[1]
                    if err.min()<min_err:
                        min_err = err.min()
                        best_f = f_i
                    f.close()
                    print(f_i+' has been read!')
                    found=True
                except:
                    found=False
                    pass
        if found and SAVE:
            print('The best file is '+best_f+' with the smallest error: {}'.format(min_err))
            savepath=fld_m+'/best_J'+str(ensbl_sz)
            if not os.path.exists(savepath): os.makedirs(savepath)
            os.system('cp '+os.path.join(fld_m,best_f)+' '+savepath)