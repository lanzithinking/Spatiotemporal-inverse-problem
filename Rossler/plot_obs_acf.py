"""
Plot autocorrelation function of spatiotemporal observations
Shiwei Lan @ ASU, 2021
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt

# model parameters
avg_traj=False
var_out='cov'

def acf(x):
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    return autocorr/autocorr.max()

# load data
fld=os.getcwd()
try:
    f=open(os.path.join(fld,'Rossler_obs_'+{True:'avg',False:'full','aug':'avgaug'}[avg_traj]+'_traj_'+{True:'nzvar',False:'','cov':'nzcov'}[var_out]+'.pckl'),'rb')
    obs,nzvar=pickle.load(f)
    f.close()
    print('Observation file has been read!')
except Exception as e:
    print(e)
    raise
obs=obs[0] # only consider single trajectory!

# compute acf's
ACF = [np.array([acf(x) for x in obs]), np.array([acf(t) for t in obs.T])]

# plot acf's
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=True,figsize=(12,5))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    plt.plot({0:np.arange(obs.shape[1])*0.3,1:np.arange(obs.shape[0])}[i], ACF[i].T, alpha=.4)
    plt.plot({0:np.arange(obs.shape[1])*0.3,1:np.arange(obs.shape[0])}[i], ACF[i].mean(0), color='r',linewidth=2)
    ax.axhline(linestyle='--',color='k',linewidth=1.5)
    ax.set_title('Autocorrelation of observations in '+{0:'space',1:'time'}[i],fontsize=16)
    ax.set_aspect('auto')
    # plt.axis([0, 1, 0, 1])
    ax.set_xlabel('distance in '+{0:'space',1:'time'}[i],fontsize=15)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
folder = './analysis'
plt.savefig(folder+'/obs_acf.png',bbox_inches='tight')
# plt.show()