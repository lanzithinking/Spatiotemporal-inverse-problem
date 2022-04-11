"""
prepare training data
Shiwei Lan @ ASU, August 2020
"""

import numpy as np

import os,sys
sys.path.append( "../" )
import pickle

TRAIN='XY'
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
    # fld_m += '/best_J'+str(ensbl_sz) # run find_best.py first
    # prepare data
    pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
    for a in range(num_algs):
        print('Working on '+algs[a]+' algorithm...')
        found=False
        # ensembles and forward outputs
        for f_i in pckl_files:
            if '_'+algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
                try:
                    f=open(os.path.join(fld_m,f_i),'rb')
                    loaded=pickle.load(f)
                    ensbl=loaded[3][:-1,:,:] if 'Y' in TRAIN else loaded[3][1:,:,:]
                    ensbl=ensbl.reshape((-1,ensbl.shape[2]))
                    fwdout=loaded[2].reshape((-1,loaded[2].shape[2]))
                    f.close()
                    print(f_i+' has been read!')
                    found=True; break
                except:
                    found=False
                    pass
        if found and SAVE:
            savepath='./train_NN/'
            if not os.path.exists(savepath): os.makedirs(savepath)
            if 'Y' in TRAIN:
                np.savez_compressed(file=os.path.join(savepath,lik_mdls[m]+'_'+algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN),X=ensbl,Y=fwdout)
            else:
                np.savez_compressed(file=os.path.join(savepath,lik_mdls[m]+'_'+algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN),X=ensbl)
    #         # how to load
    #         loaded=np.load(file=os.path.join(savepath,lik_mdls[m]+'_'+algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+'.npz'))
    #         X=loaded['X']
    #         Y=loaded['Y']