"""
Plot relative error of mean (rem) in EnK algorithms for uncertainty field u in chaotic inverse problem.
Shiwei Lan @ ASU, 2022
----------------------
"""

import os,pickle
import numpy as np


# seed=2021
# truth
true_param = list({'a':0.2, 'b':0.2, 'c':5.7}.values())

# algorithms and settings
algs=('EKI','EKS')
num_algs=len(algs)
lik_mdls=('simple','STlik')
mdl_names=('time-average','STGP')
num_mdls=len(lik_mdls)
ensbl_sz=500
configs=('Tinit','T')
num_cfgs=len(configs)
times=[10*(j+1) for j in range(10)]
num_times=len(times)
# store results
rems_spins=np.zeros((num_mdls,num_algs,num_times,10))
rems_avgs=np.zeros((num_mdls,num_algs,num_times,10))
# obtain estimates
folder = './analysis_spinavgs'
for m in range(num_mdls):
    for c in configs:
        print('Processing '+lik_mdls[m]+' likelihood model for '+{'Tinit':'spin-ups','T':'average-lengths'}[c]+'...\n')
        fld_mc = folder+'/'+lik_mdls[m]+'_'+c
        # preparation for estimates
        npz_files=[f for f in os.listdir(fld_mc) if f.endswith('.npz')] if os.path.exists(fld_mc) else []
        for i in range(num_algs):
            print('Getting estimates for '+algs[i]+' algorithm with '+str(num_times)+' '+{'Tinit':'spin-ups','T':'average-lengths'}[c]+'...')
            for j in range(num_times):
                clbl_j = {'Tinit':'_Tinit'+str(times[j])+'_T100_','T':'_Tinit100_T'+str(times[j])+'_'}[c]
                # record the rems
                num_read=0
                for f_i in npz_files:
                    if np.all([k in f_i for k in ('_'+algs[i]+'_ensbl'+str(ensbl_sz), clbl_j)]):
                        try:
                            loaded=np.load(os.path.join(fld_mc,f_i))
                            u_est=loaded['arr_0']; err=loaded['arr_1']
                            param_est=u_est[np.argmin(err[err!=0])-1]
                            if c=='Tinit':
                                rems_spins[m,i,j,num_read]=np.linalg.norm(np.exp(param_est)-true_param)/np.linalg.norm(true_param)
                            elif c=='T':
                                rems_avgs[m,i,j,num_read]=np.linalg.norm(np.exp(param_est)-true_param)/np.linalg.norm(true_param)
                            num_read+=1
                            print(f_i+' has been read!')
                        except:
                            pass
                print('%d experiment(s) have been processed for %s likelihood model with %s algorithm in time setting %d.' % (num_read, lik_mdls[m], algs[i], j))

# plot
import matplotlib.pyplot as plt
# error bar plot
fig,axes = plt.subplots(nrows=num_cfgs,ncols=num_algs,sharex=False,sharey=False,figsize=(15,12))
for i,ax in enumerate(axes.flat):
    for j in range(num_mdls):
        rems={0:rems_spins,1:rems_avgs}[i//num_cfgs]
        m=np.nanmean(rems[j,i%num_algs,:,:],axis=-1)
        s=np.nanstd(rems[j,i%num_algs,:,:],axis=-1)
        ax.plot(times, m,linestyle='-')
        ax.fill_between(times,m-1.96*s,m+1.96*s,alpha=.1*(j+1))
        # m,l,u=np.nanquantile(rems[j,i%num_algs,:,:],q=[.5,.025,.975],axis=-1)
        # ax.semilogy(times, m,linestyle='-')
        # ax.fill_between(times,l,u,alpha=.1*(j+1))
    ax.set_title(algs[i%num_algs],fontsize=18)
    ax.set_xlabel({0:'spin-up',1:'window size'}[i//num_cfgs],fontsize=16)
    ax.set_ylabel('relative error of estimate',fontsize=16)
    leg=ax.legend(mdl_names, fontsize=15, frameon=False)
plt.subplots_adjust(wspace=0.2, hspace=0.25)
# save plot
# fig.tight_layout()
# folder = './analysis'
plt.savefig(folder+'/enk_rem_spinavgs.png',bbox_inches='tight')
