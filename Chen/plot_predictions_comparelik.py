"""
Plot prediction of spatiotemporal observations in Chen inverse problem.
Shiwei Lan, February 2022 @ ASU
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

from Chen import Chen
# import sys
# sys.path.append( "../" )
STATE=0; PARAMETER=1

seed=2021
# define the inverse problem
num_traj = 1 # only consider single trajectory!
t_init = 100
t_final = 115
time_res = 150
pred_times = np.linspace(t_init, t_final, time_res)
avg_traj = False
var_out = 'cov' # True; 'cov'; False
chn_pred = Chen(num_traj=num_traj, obs_times=pred_times, avg_traj=avg_traj, var_out=var_out, seed=seed,
                use_saved_obs=False, save_obs=False)

# obtain true trajectories
true_trj=chn_pred.misfit.observe().squeeze()
# selective locations to aggregate difference in observations
cond =[t<=110 for t in chn_pred.misfit.obs_times]
obs_idx = np.where(cond)[0]

# algorithms and settings
algs=('EKI','EKS')
num_algs=len(algs)
lik_mdls=('simple','STlik')
mdl_names=('time-average','STGP')
num_mdls=len(lik_mdls)
ensbl_szs=[50,100,500,1000]
num_ensbls=len(ensbl_szs)
# store results
pred_m=np.zeros((num_mdls,num_algs,num_ensbls,chn_pred.x[STATE].shape[-1],len(chn_pred.misfit.obs_times)))
pred_std=np.zeros((num_mdls,num_algs,num_ensbls,chn_pred.x[STATE].shape[-1],len(chn_pred.misfit.obs_times)))
err_m=np.zeros((num_mdls,num_algs,num_ensbls,len(chn_pred.misfit.obs_times)))
err_std=np.zeros((num_mdls,num_algs,num_ensbls,len(chn_pred.misfit.obs_times)))
# obtain predictions
folder = './analysis'
if os.path.exists(os.path.join(folder,'predictions.npz')):
    loaded = np.load(file=os.path.join(folder,'predictions.npz'))
    pred_m, pred_std, err_m, err_std=list(map(loaded.get,['pred_m','pred_std','err_m','err_std']))
    print('Prediction data loaded!')
else:
    for m in range(num_mdls):
        print('Processing '+lik_mdls[m]+' likelihood model...\n')
        fld_m = folder+'/'+lik_mdls[m]
        # calculate predictions
        pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
        for i in range(num_algs):
            for j in range(num_ensbls):
                print('Getting estimates for '+algs[i]+' algorithm with '+str(ensbl_szs[j])+' ensembles...')
                found=False
                fwdout_mean=0; fwdout_std=0
                fwderr_mean=0; fwderr_std=0
        #         num_read=0
                for f_i in pckl_files:
                    if '_'+algs[i]+'_ensbl'+str(ensbl_szs[j])+'_' in f_i:
                        try:
                            f=open(os.path.join(fld_m,f_i),'rb')
                            f_read=pickle.load(f)
                            err,ensbls=f_read[1],f_read[3]
                            params=ensbls[np.argmin(err[err!=0])]
                            fwdout_mean=0; fwdout_std=0; fwderr_mean=0; fwderr_std=0; num_read=0
                            # prog=np.ceil(ensbl_szs[j]*(.1+np.arange(0,1,.1)))
                            for s in range(ensbl_szs[j]):
                                # if s+1 in prog:
                                #     print('{0:.0f}% has been completed.'.format(np.float(s+1)/ensbl_szs[j]*100))
                                chn_pred.x[PARAMETER]=np.exp(params[s]);
                                chn_pred.x[STATE]=chn_pred.ode.solveFwd(params=chn_pred.x[PARAMETER], t=chn_pred.misfit.obs_times)[0]
                                pred=chn_pred.misfit.observe(sol=chn_pred.x[STATE]).squeeze()
                                fwdout_mean+=pred/ensbl_szs[j]
                                fwdout_std+=pred**2/ensbl_szs[j]
                                err=np.linalg.norm(pred-true_trj,axis=1)
                                fwderr_mean+=err/ensbl_szs[j]
                                fwderr_std+=err**2/ensbl_szs[j]
        #                         num_read+=1
                            f.close()
                            print(f_i+' has been read!')
                            f_read=f_i
                            found=True; break
                        except:
                            pass
                if found:
        #             fwdout_mean=fwdout_mean/num_read; fwdout_std=fwdout_std/num_read
                    pred_m[m][i][j]=fwdout_mean.T
                    pred_std[m][i][j]=np.sqrt(fwdout_std - fwdout_mean**2).T
                    err_m[m][i][j]=fwderr_mean.T
                    err_std[m][i][j]=np.sqrt(fwderr_std - fwderr_mean**2).T
    # save
    np.savez_compressed(file=os.path.join(folder,'predictions'), pred_m=pred_m, pred_std=pred_std, err_m=err_m, err_std=err_std)

# plot prediction
# num_algs-=1
plt.rcParams['image.cmap'] = 'jet'

# all locations
num_rows=2
j=2
for k in range(chn_pred.x[STATE].shape[-1]):
    fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((2*num_algs)/num_rows)),sharex=True,sharey=True,figsize=(11,8))
    # k = 50 # select location to plot observations
    for ii,ax in enumerate(axes.flat):
        m=ii//num_algs; i=ii%num_algs
        ax.plot(chn_pred.misfit.obs_times,true_trj[:,k],color='red',linewidth=1.5)
        ax.plot(chn_pred.misfit.obs_times,pred_m[m][i][j][k],linestyle='--')
        ax.fill_between(chn_pred.misfit.obs_times,pred_m[m][i][j][k]-1.96*pred_std[m][i][j][k],pred_m[m][i][j][k]+1.96*pred_std[m][i][j][k],color='b',alpha=.1)
        ax.set_title(mdl_names[m]+' - '+algs[i])
        ax.set_aspect('auto')
        # plt.axis([0, 1, 0, 1])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # save plot
    # fig.tight_layout()
    if not os.path.exists(folder+'/predictions'): os.makedirs(folder+'/predictions')
    plt.savefig(folder+'/predictions/comparelik_'+('x','y','z')[k]+'.png',bbox_inches='tight')
    # plt.show()

# num_rows=2
# fig,axes = plt.subplots(nrows=num_rows,ncols=2,sharex=False,sharey=False,figsize=(9,8))
num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=3,sharex=False,sharey=False,figsize=(18,5))
i=1; j=2;
for ii,ax in enumerate(axes.flat):
    plt.axes(axes.flat[ii])
    # if ii==0:
    #     cred_cover = np.mean(np.logical_and(pred_m[0][i][j]-1.96*pred_std[0][i][j] < true_trj.T, true_trj.T < pred_m[0][i][j]+1.96*pred_std[0][i][j]),axis=0)
    #     ax.plot(chn_pred.misfit.obs_times,cred_cover,linestyle='--')
    #     cred_cover = np.mean(np.logical_and(pred_m[1][i][j]-1.96*pred_std[1][i][j] < true_trj.T, true_trj.T < pred_m[1][i][j]+1.96*pred_std[1][i][j]),axis=0)
    #     ax.plot(chn_pred.misfit.obs_times,cred_cover,linestyle='-.')
    #     ax.set_xlabel('t', fontsize=16)
    #     ax.set_title('truth covering rate of credible bands',fontsize=16)
    #     plt.legend(mdl_names,frameon=False, fontsize=15)
    # # elif ii==1:
    # #     ax.plot(chn_pred.misfit.obs_times,np.zeros(len(chn_pred.misfit.obs_times)),color='red',linewidth=1.5)
    # #     ax.plot(chn_pred.misfit.obs_times,err_m[0][i][j],linestyle='--')
    # #     ax.fill_between(chn_pred.misfit.obs_times,err_m[0][i][j]-1.96*err_std[0][i][j],err_m[0][i][j]+1.96*err_std[0][i][j],color='b',alpha=.1)
    # #     ax.plot(chn_pred.misfit.obs_times,err_m[1][i][j],linestyle='-.')
    # #     ax.fill_between(chn_pred.misfit.obs_times,err_m[1][i][j]-1.96*err_std[1][i][j],err_m[1][i][j]+1.96*err_std[1][i][j],color='y',alpha=.1)
    # else:
    #     # m=ii%2; k=locs[m]
    #     k=ii-1
    k=ii
    ax.plot(chn_pred.misfit.obs_times,true_trj[:,k],color='red')
    ax.plot(chn_pred.misfit.obs_times,pred_m[0][i][j][k],linestyle='--')
    ax.fill_between(chn_pred.misfit.obs_times,pred_m[0][i][j][k]-1.96*pred_std[0][i][j][k],pred_m[0][i][j][k]+1.96*pred_std[0][i][j][k],color='b',alpha=.1)
    ax.plot(chn_pred.misfit.obs_times,pred_m[1][i][j][k],linestyle='-.',color='g')
    ax.fill_between(chn_pred.misfit.obs_times,pred_m[1][i][j][k]-1.96*pred_std[1][i][j][k],pred_m[1][i][j][k]+1.96*pred_std[1][i][j][k],color='g',alpha=.2)
    ax.set_xlabel('t', fontsize=16)
    ax.set_title('forward prediction ('+('x','y','z')[k]+')',fontsize=16)
    plt.legend(('truth',)+mdl_names,frameon=False, fontsize=15)
    ax.set_aspect('auto')
    # plt.axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.15, hspace=0.1)
# save plot
# fig.tight_layout()
# if not os.path.exists(folder+'/predictions'): os.makedirs(folder+'/predictions')
plt.savefig(folder+'/predictions_comparelik.png',bbox_inches='tight')
# plt.show()