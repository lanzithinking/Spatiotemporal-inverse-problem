"""
prepare training data
Shiwei Lan @ ASU, August 2020
"""

import numpy as np
import os,sys
import pickle

# the inverse problem


TRAIN={0:'XimgY',1:'XY'}[1]

np.random.seed(2020)

# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
# preparation for estimates
folder = './result'
pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
ensbl_sz=500
max_iter=10
img_out=('img' in TRAIN)

PLOT=False
SAVE=True
# prepare data
for a in range(num_algs):
    for j in ['_full_','_avgaug_']:
        print('Working on '+algs[a]+' algorithm...')
        found=False
        # ensembles and forward outputs
        for f_i in pckl_files:
            if j+algs[a]+'_ensbl'+str(ensbl_sz)+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
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
        if found:
            if img_out and PLOT:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(8,8), facecolor='white')
                ax = fig.add_subplot(111, frameon=False)
                plt.ion()
                plt.show(block=False)
                for t in range(ensbl.shape[0]):
                    plt.cla()
                    plt.imshow(ensbl[t],origin='lower',extent=[0,1,0,1])
                    plt.title('Ensemble {}'.format(t))
                    plt.show()
                    plt.pause(1.0/100.0)
            if SAVE:
                savepath='./train_NN/'
                os.makedirs(savepath, exist_ok=True)
                if 'Y' in TRAIN:
                    if '_full_' in j:
                        np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+'_stlik'),X=ensbl,Y=fwdout)
                    else:
                        np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN),X=ensbl,Y=fwdout)
                else:
                    np.savez_compressed(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN),X=ensbl)
        #         # how to load
        #         loaded=np.load(file=os.path.join(savepath,algs[a]+'_ensbl'+str(ensbl_sz)+'_training_'+TRAIN+'.npz'))
        #         X=loaded['X']
        #         Y=loaded['Y']