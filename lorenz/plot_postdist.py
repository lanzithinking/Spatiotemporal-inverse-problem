"""
Plot posterior distributions
-----------------------------------------
Shiwei Lan @ ASU 2020
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# likelihood models
mdls=('simple','STlik')
n_mdl=len(mdls)
mdl_names=('time-average','STGP')
# training data algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1
# load data
ensbl_sz = 500

# data folder
folder='./analysis'
pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
num_samp=10000; input_dim=3
samps=np.zeros((n_mdl,num_samp,input_dim))

sns.set(font_scale=1.1)
for m in range(n_mdl):
    print('Working on '+mdls[m]+' model...\n')
    avg_traj = {'simple':'aug','STlik':False}[mdls[m]] # True; 'aug'; False
    STlik = (mdls[m]=='STlik')
    lbl = {True:'avg',False:'full','aug':'avgaug'}[avg_traj]
    found=False
    for f_i in pckl_files:
        if '_'+lbl+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f)
                samp=f_read[3]
                f.close()
                print(f_i+' has been read!')
                found=True; break
            except:
                pass
    if found:
        # get posterior samples
        samps[m]=np.exp(samp)
        # get training ensembles
        loaded=np.load(file=os.path.join('./train_NN/',mdls[m]+'_'+algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
        X=loaded['X']
        ensbls=pd.DataFrame(np.exp(X[np.random.RandomState(2021).choice(np.arange(ensbl_sz,X.shape[0]),size=num_samp,replace=False)]),columns=['$\\sigma$','$\\beta$','$\\rho$'])
        
        # form the data frame
        mdl_array=np.array([mdl_names[m]]*num_samp)
        df_samps=pd.DataFrame(samps[m].reshape((-1,input_dim)),columns=['$\\sigma$','$\\beta$','$\\rho$'])
        df_samps['model']=mdl_array
        
        g=sns.PairGrid(df_samps,diag_sharey=False)
        g.map_upper(plt.scatter,s=1,alpha=0.5)
        # g.map_lower(sns.kdeplot)
        def pairkde(x, y, **kwargs):
            ax=plt.gca()
            pts=kwargs.pop('pts',ensbls)
            ax.scatter(pts[x.name],pts[y.name],**kwargs)
            sns.kdeplot(x,y)
        g.map_lower(pairkde, s=1,alpha=.5,color='red')
        g.map_diag(sns.kdeplot,lw=2)
        g.add_legend()
        for ax in g.axes.flatten():
            if ax:
                # rotate y axis labels
                ax.set_ylabel(ax.get_ylabel(), rotation = 0)
                # set y labels alignment
                ax.yaxis.get_label().set_horizontalalignment('right')
        # g.fig.suptitle('MCMC')
        # g.fig.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(folder,'postdist_'+mdl_names[m]+'.png'),bbox_inches='tight')