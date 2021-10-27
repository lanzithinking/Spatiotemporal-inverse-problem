"""
Plot prediction of spatiotemporal observations in Advection-Diffusion inverse problem.
Shiwei Lan, October 2021 @ ASU
"""

import os,pickle
import numpy as np
import dolfin as df
# import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mp

from advdiff import advdiff
import sys
sys.path.append( "../" )
# from nn.ae import AutoEncoder
# from nn.cae import ConvAutoEncoder
# from nn.vae import VAE
# from tensorflow.keras.models import load_model
from util.dolfin_gadget import *
from util.multivector import *
STATE=0; PARAMETER=1

# # functions needed to make even image size
# def pad(A,width=[1]):
#     shape=A.shape
#     if len(width)==1: width=np.tile(width,len(shape))
#     if not any(width): return A
#     assert len(width)==len(shape), 'non-matching padding width!'
#     pad_width=tuple((0,i) for i in width)
#     return np.pad(A, pad_width)
# def chop(A,width=[1]):
#     shape=A.shape
#     if len(width)==1: width=np.tile(width,len(shape))
#     if not any(width): return A
#     assert len(width)==len(shape), 'non-matching chopping width!'
#     chop_slice=tuple(slice(0,-i) for i in width)
#     return A[chop_slice]

seed=2020
# define the inverse problem
meshsz = (61,61)
eldeg = 1
gamma = 2.; delta = 10.
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
# adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# # set up latent
# meshsz_latent = (21,21)
# adif_latent = advdiff(mesh=meshsz_latent, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
# adif_latent.prior.V=adif_latent.prior.Vh

# modify the problem for prediction
simulation_times = np.arange(0., 5.+.5*.1, .1)
observation_times = simulation_times
adif_pred = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise,
                    simulation_times=simulation_times, observation_times=observation_times, save_obs=False, seed=seed)
# obtain true trajectories
ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
true_param = df.interpolate(ic_expr, adif.prior.V).vector()
adif_pred.x[PARAMETER]=true_param; adif_pred.pde.solveFwd(adif_pred.x[STATE],adif_pred.x)
true_trj=adif_pred.misfit.observe(adif_pred.x)

# selective locations to aggregate difference in observations
cond = np.logical_or([abs(x-.25)<.01 or abs(x-.6)<.01 for x in adif.misfit.targets[:,0]],[abs(y-.4)<.01 or abs(y-.85)<.01 for y in adif.misfit.targets[:,1]]) # or abs(y-.85)<.01
slab_idx = np.where(cond)[0]

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

# ##------ define networks ------##
# # training data algorithms
# algs=['EKI','EKS']
# num_algs=len(algs)
# alg_no=1
# # load data
# ensbl_sz = 500
# folder = './train_NN_eldeg'+str(eldeg)
#
# ##---- AUTOENCODER ----##
# AE={0:'ae',1:'cae',2:'vae'}[0]
# # prepare for training data
# if 'c' in AE:
#     loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XimgY.npz'))
#     X=loaded['X']
#     X=X[:,:-1,:-1,None]
# else :
#     loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
#     X=loaded['X']
# num_samp=X.shape[0]
# # n_tr=np.int(num_samp*.75)
# # x_train=X[:n_tr]
# # x_test=X[n_tr:]
# tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
# te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
# x_train,x_test=X[tr_idx],X[te_idx]
# # define autoencoder
# if AE=='ae':
#     half_depth=3; latent_dim=adif_latent.prior.V.dim()
#     droprate=0.
#     activation='elu'
# #     activation=tf.keras.layers.LeakyReLU(alpha=1.5)
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
#     lambda_=0.
#     autoencoder=AutoEncoder(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, droprate=droprate,
#                             activation=activation, optimizer=optimizer)
# elif AE=='cae':
#     num_filters=[16,8]; latent_dim=adif_latent.prior.V.dim()
# #     activations={'conv':tf.keras.layers.LeakyReLU(alpha=0.1),'latent':None} # [16,1]
#     activations={'conv':'elu','latent':'linear'}
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
#     autoencoder=ConvAutoEncoder(x_train.shape[1:], num_filters=num_filters, latent_dim=latent_dim,
#                                 activations=activations, optimizer=optimizer)
# elif AE=='vae':
#         half_depth=5; latent_dim=adif_latent.prior.V.dim()
#         repatr_out=False; beta=1.
#         activation='elu'
# #         activation=tf.keras.layers.LeakyReLU(alpha=0.01)
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
#         autoencoder=VAE(x_train.shape[1], half_depth=half_depth, latent_dim=latent_dim, repatr_out=repatr_out,
#                         activation=activation, optimizer=optimizer, beta=beta)
# f_name=[AE+'_'+i+'_'+algs[alg_no]+str(ensbl_sz) for i in ('fullmodel','encoder','decoder')]
# # load autoencoder
# try:
#     autoencoder.model=load_model(os.path.join(folder,f_name[0]+'.h5'),custom_objects={'loss':None})
#     print(f_name[0]+' has been loaded!')
#     autoencoder.encoder=load_model(os.path.join(folder,f_name[1]+'.h5'),custom_objects={'loss':None})
#     print(f_name[1]+' has been loaded!')
#     autoencoder.decoder=load_model(os.path.join(folder,f_name[2]+'.h5'),custom_objects={'loss':None})
#     print(f_name[2]+' has been loaded!')
# except:
#     print('\nNo autoencoder found. Training {}...\n'.format(AE))
#     epochs=200
#     patience=0
#     noise=0.
#     kwargs={'patience':patience}
#     if AE=='ae' and noise: kwargs['noise']=noise
#     autoencoder.train(x_train,x_test=x_test,epochs=epochs,batch_size=64,verbose=1,**kwargs)
#     # save autoencoder
#     autoencoder.model.save(os.path.join(folder,f_name[0]+'.h5'))
#     autoencoder.encoder.save(os.path.join(folder,f_name[1]+'.h5'))
#     autoencoder.decoder.save(os.path.join(folder,f_name[2]+'.h5'))

# algorithms
algs=('pCN','infMALA','infHMC')#,'epCN','einfMALA','einfHMC','DREAMpCN','DREAMinfMALA','DREAMinfHMC','DRinfmHMC')
alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','e-pCN','e-$\infty$-MALA','e-$\infty$-HMC','DREAM-pCN','DREAM-$\infty$-MALA','DREAM-$\infty$-HMC','DR-$\infty$-HMC')
num_algs=len(algs)
lik_mdls=('simple','STlik')
num_mdls=len(lik_mdls)
# obtain estimates
folder = './analysis_eldeg'+str(eldeg)
pred_m=[np.zeros((num_algs,adif_pred.misfit.targets.shape[0],len(adif_pred.misfit.observation_times))),np.zeros((num_algs,adif_pred.misfit.targets.shape[0],len(adif_pred.misfit.observation_times)))]
pred_std=[np.zeros((num_algs,adif_pred.misfit.targets.shape[0],len(adif_pred.misfit.observation_times))),np.zeros((num_algs,adif_pred.misfit.targets.shape[0],len(adif_pred.misfit.observation_times)))]
err_m=[np.zeros((num_algs,len(adif_pred.misfit.observation_times))),np.zeros((num_algs,len(adif_pred.misfit.observation_times)))]
err_std=[np.zeros((num_algs,len(adif_pred.misfit.observation_times))),np.zeros((num_algs,len(adif_pred.misfit.observation_times)))]
if os.path.exists(os.path.join(folder,'predictions.npz')):
    loaded = np.load(file=os.path.join(folder,'predictions.npz'))
    pred_m, pred_std, err_m, err_std=list(map(loaded.get,['pred_m','pred_std','err_m','err_std']))
    print('Prediction data loaded!')
else:
    for m in range(num_mdls):
        print('Processing '+lik_mdls[m]+' likelihood model...\n')
        fld_m = folder+'/'+lik_mdls[m]
        # preparation for estimates
        hdf5_files=[f for f in os.listdir(fld_m) if f.endswith('.h5')]
        pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
        num_samp=5000
        prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
        for i in range(num_algs):
            print('Getting estimates for '+algs[i]+' algorithm...')
            # obtain weights
            wts=np.ones(num_samp)/num_samp
            if 'DREAM' in algs[i]:
                for f_i in pckl_files:
                    if '_'+algs[i]+'_' in f_i:
                        try:
                            f=open(os.path.join(fld_m,f_i),'rb')
                            f_read=pickle.load(f)
                            logwts=f_read[4]; logwts-=logwts.max()
                            wts=np.exp(logwts); wts/=np.sum(wts)
                            f.close()
                            print(f_i+' has been read!')
                        except:
                            pass
            bip=adif_latent if 'DREAM' in algs[i] else adif
            # calculate posterior estimates
            found=False
            samp_f=df.Function(bip.prior.V,name="parameter")
            fwdout_mean=0; fwdout_std=0
            fwderr_mean=0; fwderr_std=0
    #         num_read=0
            for f_i in hdf5_files:
                if '_'+algs[i]+'_' in f_i:
                    try:
                        f=df.HDF5File(bip.pde.mpi_comm,os.path.join(fld_m,f_i),"r")
                        fwdout_mean=0; fwdout_std=0; fwderr_mean=0; fwderr_std=0; num_read=0
                        for s in range(num_samp):
                            if s+1 in prog:
                                print('{0:.0f}% has been completed.'.format(np.float(s+1)/num_samp*100))
                            f.read(samp_f,'sample_{0}'.format(s))
                            u=samp_f.vector()
                            if '_whitened_latent' in f_i: u=bip.prior.v2u(u)
                            if 'DREAM' in algs[i]:
                                if 'c' in AE:
                                    u_latin=adif_latent.vec2img(u)
                                    width=tuple(np.mod(i,2) for i in u_latin.shape)
                                    u_latin=chop(u_latin,width)[None,:,:,None] if autoencoder.activations['latent'] is None else u_latin.flatten()[None,:]
                                    u=adif.img2vec(pad(np.squeeze(autoencoder.decode(u_latin)),width),adif.prior.V if eldeg>1 else None)
                                else:
                                    u_latin=u.get_local()[None,:]
                                    u_decoded=autoencoder.decode(u_latin).flatten()
                                    u=adif.prior.gen_vector(u_decoded) if eldeg==1 else vinPn(u_decoded, adif.prior.V)
    #                         else:
    #                             u=u_
                            if '_whitened_emulated' in f_i: u=adif.prior.v2u(u)
                            adif_pred.x[PARAMETER]=u; adif_pred.pde.solveFwd(adif_pred.x[STATE],adif_pred.x)
                            pred=adif_pred.misfit.observe(adif_pred.x)
                            fwdout_mean+=wts[s]*pred
                            fwdout_std+=wts[s]*pred**2
                            err=np.linalg.norm(pred[:,slab_idx]-true_trj[:,slab_idx],axis=1)
                            fwderr_mean+=wts[s]*err
                            fwderr_std+=wts[s]*err**2
    #                         num_read+=1
                        f.close()
                        print(f_i+' has been read!')
                        f_read=f_i
                        found=True; break
                    except:
                        pass
            if found:
    #             fwdout_mean=fwdout_mean/num_read; fwdout_std=fwdout_std/num_read
                pred_m[m][i]=fwdout_mean.T
                pred_std[m][i]=np.sqrt(fwdout_std - fwdout_mean**2).T
                err_m[m][i]=fwderr_mean.T
                err_std[m][i]=np.sqrt(fwderr_std - fwderr_mean**2).T
    # save
    np.savez_compressed(file=os.path.join(folder,'predictions'), pred_m=pred_m, pred_std=pred_std, err_m=err_m, err_std=err_std)

# plot prediction
# num_algs-=1
plt.rcParams['image.cmap'] = 'jet'

# all locations
# num_rows=2
# for k in range(adif.misfit.targets.shape[0]):
#     fig,axes = plt.subplots(nrows=num_rows,ncols=np.int(np.ceil((2*num_algs)/num_rows)),sharex=True,sharey=True,figsize=(11,8))
#     # k = 50 # select location to plot observations
#     for i,ax in enumerate(axes.flat):
#         m=i//num_algs; j=i%num_algs
#         ax.plot(adif_pred.misfit.observation_times,true_trj[:,k],color='red',linewidth=1.5)
#         ax.scatter(adif.misfit.observation_times,obs[:,k])
#         ax.plot(adif_pred.misfit.observation_times,pred_m[m][j][k],linestyle='--')
#         ax.fill_between(adif_pred.misfit.observation_times,pred_m[m][j][k]-1.96*pred_std[m][j][k],pred_m[m][j][k]+1.96*pred_std[m][j][k],color='b',alpha=.1)
#         ax.set_title(alg_names[i%axes.shape[1]])
#         ax.set_aspect('auto')
#         # plt.axis([0, 1, 0, 1])
#     plt.suptitle('x=%.3f, \t y=%.3f'% tuple(adif.misfit.targets[k]))
#     plt.subplots_adjust(wspace=0.1, hspace=0.2)
#     # save plot
#     # fig.tight_layout()
#     if not os.path.exists(folder+'/predictions'): os.makedirs(folder+'/predictions')
#     plt.savefig(folder+'/predictions/comparelik_k'+str(k)+'.png',bbox_inches='tight')
#     # plt.show()

# selective locations
# num_rows=3
# fig,axes = plt.subplots(nrows=num_rows,ncols=num_algs,sharex=True,sharey=False,figsize=(11,10))
# locs=[16,26]
# for i,ax in enumerate(axes.flat):
#     m=i//num_algs
#     if m<num_rows-1:
#         j=i%num_algs; k=locs[m]
#         ax.plot(adif_pred.misfit.observation_times,true_trj[:,k],color='red',linewidth=1.5)
#         ax.scatter(adif.misfit.observation_times,obs[:,k])
#         ax.plot(adif_pred.misfit.observation_times,pred_m[0][j][k],linestyle='--')
#         ax.fill_between(adif_pred.misfit.observation_times,pred_m[0][j][k]-1.96*pred_std[0][j][k],pred_m[0][j][k]+1.96*pred_std[0][j][k],color='b',alpha=.1)
#         ax.plot(adif_pred.misfit.observation_times,pred_m[1][j][k],linestyle='-.')
#         ax.fill_between(adif_pred.misfit.observation_times,pred_m[1][j][k]-1.96*pred_std[1][j][k],pred_m[1][j][k]+1.96*pred_std[1][j][k],color='y',alpha=.1)
#     else:
#         ax.plot(adif_pred.misfit.observation_times,np.zeros(len(adif_pred.misfit.observation_times)),color='red',linewidth=1.5)
#         ax.plot(adif_pred.misfit.observation_times,np.linalg.norm(pred_m[0][j]-true_trj.T,axis=0),linestyle='--')
#         # ax.fill_between(adif_pred.misfit.observation_times,pred_m[0][j][k]-1.96*pred_std[0][j][k],pred_m[0][j][k]+1.96*pred_std[0][j][k],color='b',alpha=.1)
#         ax.plot(adif_pred.misfit.observation_times,np.linalg.norm(pred_m[1][j]-true_trj.T,axis=0),linestyle='-.')
#         # ax.fill_between(adif_pred.misfit.observation_times,pred_m[1][j][k]-1.96*pred_std[1][j][k],pred_m[1][j][k]+1.96*pred_std[1][j][k],color='g',alpha=.1)
#     ax.set_title(alg_names[i%axes.shape[1]]+' (x=%.3f, y=%.3f)'% tuple(adif.misfit.targets[k]))
#     ax.set_aspect('auto')
#     # plt.axis([0, 1, 0, 1])
# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# # save plot
# # fig.tight_layout()
# if not os.path.exists(folder+'/predictions'): os.makedirs(folder+'/predictions')
# plt.savefig(folder+'/predictions_comparelik_0.png',bbox_inches='tight')
# plt.show()

# num_rows=2
# fig,axes = plt.subplots(nrows=num_rows,ncols=2,sharex=False,sharey=False,figsize=(9,8))
num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=3,sharex=False,sharey=False,figsize=(14,4))
locs=[16,26]
j=2;
for i,ax in enumerate(axes.flat):
    if i==0:
        plt.axes(axes.flat[i])
    #     df.plot(adif_pred.misfit.Vh.mesh())
    #     ax.scatter(adif_pred.misfit.targets[slab_idx,0],adif_pred.misfit.targets[slab_idx,1])
        cred_cover = np.mean(np.logical_and(pred_m[0][j]-1.96*pred_std[0][j] < true_trj.T, true_trj.T < pred_m[0][j]+1.96*pred_std[0][j]),axis=0)
        ax.plot(adif_pred.misfit.observation_times,cred_cover,linestyle='--')
        cred_cover = np.mean(np.logical_and(pred_m[1][j]-1.96*pred_std[1][j] < true_trj.T, true_trj.T < pred_m[1][j]+1.96*pred_std[1][j]),axis=0)
        ax.plot(adif_pred.misfit.observation_times,cred_cover,linestyle='-.')
        ax.set_xlabel('t')
        ax.set_title('truth covering rate of credible bands')
        plt.legend(['simple likelihood','spatiotemporal likelihood'],frameon=False)
    # elif i==1:
    #     ax.plot(adif_pred.misfit.observation_times,np.zeros(len(adif_pred.misfit.observation_times)),color='red',linewidth=1.5)
    #     ax.plot(adif_pred.misfit.observation_times,err_m[0][j],linestyle='--')
    #     ax.fill_between(adif_pred.misfit.observation_times,err_m[0][j]-1.96*err_std[0][j],err_m[0][j]+1.96*err_std[0][j],color='b',alpha=.1)
    #     ax.plot(adif_pred.misfit.observation_times,err_m[1][j],linestyle='-.')
    #     ax.fill_between(adif_pred.misfit.observation_times,err_m[1][j]-1.96*err_std[1][j],err_m[1][j]+1.96*err_std[1][j],color='y',alpha=.1)
    else:
        # m=i%2; k=locs[m]
        k=locs[i-1]
        ax.plot(adif_pred.misfit.observation_times,true_trj[:,k],color='red',linewidth=1.5)
        ax.scatter(adif.misfit.observation_times,obs[:,k])
        ax.plot(adif_pred.misfit.observation_times,pred_m[0][j][k],linestyle='--')
        ax.fill_between(adif_pred.misfit.observation_times,pred_m[0][j][k]-1.96*pred_std[0][j][k],pred_m[0][j][k]+1.96*pred_std[0][j][k],color='b',alpha=.1)
        ax.plot(adif_pred.misfit.observation_times,pred_m[1][j][k],linestyle='-.')
        ax.fill_between(adif_pred.misfit.observation_times,pred_m[1][j][k]-1.96*pred_std[1][j][k],pred_m[1][j][k]+1.96*pred_std[1][j][k],color='y',alpha=.2)
        ax.set_xlabel('t')
        ax.set_title('forward prediction (x=%.3f, y=%.3f)'% tuple(adif.misfit.targets[k]))
    ax.set_aspect('auto')
    # plt.axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.2, hspace=0.2)
# save plot
# fig.tight_layout()
if not os.path.exists(folder+'/predictions'): os.makedirs(folder+'/predictions')
plt.savefig(folder+'/predictions_comparelik.png',bbox_inches='tight')
# plt.show()