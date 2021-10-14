"""
Plot autocorrelation function of spatiotemporal observations
Shiwei Lan @ ASU, 2021
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt


def acf(x):
    autocorr = np.correlate(x, x, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    return autocorr/autocorr.max()

# load data
fld=os.getcwd()
try:
    f=open(os.path.join(fld,'AdvDiff_obs.pckl'),'rb')
    obs,noise_variance=pickle.load(f)
    f.close()
    print('Observation file has been read!')
except Exception as e:
    print(e)
    raise

# compute acf's
ACF = [np.array([acf(x) for x in obs]), np.array([acf(t) for t in obs.T])]

# plot acf's
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=True,figsize=(12,5))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    plt.plot({0:np.arange(obs.shape[1])*0.025,1:np.arange(obs.shape[0])*0.2}[i], ACF[i].T, alpha=.4)
    plt.plot({0:np.arange(obs.shape[1])*0.025,1:np.arange(obs.shape[0])*0.2}[i], ACF[i].mean(0), color='r',linewidth=2)
    ax.axhline(linestyle='--',color='k',linewidth=1.5)
    ax.set_title('Autocorrelation of observations in '+{0:'space',1:'time'}[i],fontsize=16)
    ax.set_aspect('auto')
    # plt.axis([0, 1, 0, 1])
    ax.set_xlabel('distance in '+{0:'space',1:'time'}[i],fontsize=15)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
eldeg=1
folder = './analysis_eldeg'+str(eldeg)
plt.savefig(folder+'/obs_acf.png',bbox_inches='tight')
# plt.show()