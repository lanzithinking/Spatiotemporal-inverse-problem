"""
Plot errors (misfit) in EnK algorithms for uncertainty field u in chaotic inverse problem.
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
num_mdls=len(lik_mdls)
ensbl_szs=[50,100,500,1000]
num_ensbls=len(ensbl_szs)
# store results
errs=np.empty((num_mdls,num_algs,num_ensbls,10,50)); errs.fill(np.nan)
argmins=np.zeros((num_mdls,num_algs,num_ensbls,10))
# obtain estimates
folder = './analysis'
for m in range(num_mdls):
    print('Processing '+lik_mdls[m]+' likelihood model...\n')
    # fld_m = folder+('_fixedhyper/' if m==0 else '/')+lik_mdls[m]
    fld_m = folder+'/'+lik_mdls[m]
    # preparation for estimates
    pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
    for i in range(num_algs):
        for j in range(num_ensbls):
            print('Getting estimates for '+algs[i]+' algorithm with '+str(ensbl_szs[j])+' ensembles...')
            # record the errors
            num_read=0
            for f_i in pckl_files:
                if '_'+algs[i]+'_ensbl'+str(ensbl_szs[j])+'_' in f_i:
                    try:
                        f=open(os.path.join(fld_m,f_i),'rb')
                        f_read=pickle.load(f)
                        err=f_read[1]
                        nz_idx=np.where(err!=0)[0]
                        errs[m,i,j,num_read,nz_idx]=err[nz_idx]
                        argmins[m,i,j,num_read]=np.argmin(err[nz_idx])-1
                        num_read+=1
                        f.close()
                        print(f_i+' has been read!')
                    except:
                        pass
            print('%d experiment(s) have been processed for %s algorithm with %d ensembles for %s likelihood model.' % (num_read, algs[i], ensbl_szs[j], lik_mdls[m]))

# plot
import matplotlib.pyplot as plt
# for j in range(num_ensbls):
#     fig,axes = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,figsize=(12,5))
#     for i,ax in enumerate(axes.flat):
#         plt.axes(ax)
#         for m in range(num_mdls):
#             plt.plot(errs[m,i,j,:,:].T, color=('red','blue')[m], alpha=.4)
#             plt.plot(argmins[m,i,j,:],[0.1]*len(argmins[m,i,j,:]), '|', color=('red','blue')[m])
#         ax.set_title(algs[i],fontsize=16)
#         ax.set_aspect('auto')
#         ax.set_xlabel('iteration',fontsize=15)
#         ax.set_ylabel('error',fontsize=15)
#         leg=ax.legend(lik_mdls)
#         leg.legendHandles[0].set_color('red')
#         leg.legendHandles[1].set_color('blue')
#     plt.subplots_adjust(wspace=0.2, hspace=0.2)
#     # save plot
#     # fig.tight_layout()
#     # folder = './analysis'
#     plt.savefig(folder+'/enk_err_J'+str(ensbl_szs[j])+'.png',bbox_inches='tight')
#     # plt.show()

# error bar plot
fig,axes = plt.subplots(nrows=num_algs,ncols=num_mdls,sharex=True,sharey=False,figsize=(15,12))
for i,ax in enumerate(axes.flat):
    for j in range(num_ensbls):
        m=np.nanmean(errs[i%num_mdls,i//num_algs,j,:,:],axis=0)
        s=np.nanstd(errs[i%num_mdls,i//num_algs,j,:,:],axis=0)
        ax.plot(m,linestyle='-')
        ax.fill_between(np.arange(len(m)),m-1.96*s,m+1.96*s,alpha=.1*(j+1))
        # m,l,u=np.nanquantile(errs[i%num_mdls,i//num_algs,j,:,:],q=[.5,.025,.975],axis=0)
        # ax.semilogy(m,linestyle='-')
        # ax.fill_between(np.arange(len(m)),l,u,alpha=.1*(j+1))
    ax.set_title(lik_mdls[i%num_mdls]+' - '+algs[i//num_algs],fontsize=16)
    ax.set_xlabel('iteration',fontsize=15)
    ax.set_ylabel('error',fontsize=15)
    leg=ax.legend(['J='+str(j) for j in ensbl_szs], ncol=num_ensbls, frameon=False)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
# save plot
# fig.tight_layout()
# folder = './analysis'
plt.savefig(folder+'/enk_err.png',bbox_inches='tight')
