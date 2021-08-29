"""
Get mean squared errors of uncertainty field u in Advection-Diffusion inverse problem.
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Modified for STIP August 2021 @ ASU
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


# functions needed to make even image size
def pad(A,width=[1]):
    shape=A.shape
    if len(width)==1: width=np.tile(width,len(shape))
    if not any(width): return A
    assert len(width)==len(shape), 'non-matching padding width!'
    pad_width=tuple((0,i) for i in width)
    return np.pad(A, pad_width)
def chop(A,width=[1]):
    shape=A.shape
    if len(width)==1: width=np.tile(width,len(shape))
    if not any(width): return A
    assert len(width)==len(shape), 'non-matching chopping width!'
    chop_slice=tuple(slice(0,-i) for i in width)
    return A[chop_slice]

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
# get the true parameter (initial condition)
ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
true_param = df.interpolate(ic_expr, adif.prior.V).vector()

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
# store results
lik_mdls=('simple','STlik')
num_mdls=len(lik_mdls)
mse_m=np.zeros((num_mdls,num_algs))
mse_s=np.zeros((num_mdls,num_algs))
# obtain estimates
folder = './analysis_eldeg'+str(eldeg)
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
        wts=[]
        for f_i in pckl_files:
            if '_'+algs[i]+'_' in f_i:
                if 'DREAM' in algs[i]:
                    try:
                        f=open(os.path.join(fld_m,f_i),'rb')
                        f_read=pickle.load(f)
                        logwts=f_read[4]; logwts-=logwts.max()
                        wts.append(np.exp(logwts)); wts[-1]/=np.sum(wts[-1])
                        f.close()
                        print(f_i+' has been read!')
                    except:
                        pass
                else:
                    wts.append(np.ones(num_samp)/num_samp)
        bip=adif_latent if 'DREAM' in algs[i] else adif
        # calculate posterior estimates
        found=False
        samp_f=df.Function(bip.prior.V,name="parameter")
        # samp_mean=adif.prior.gen_vector(); samp_mean.zero()
        # samp_std=adif.prior.gen_vector(); samp_std.zero()
        errs=[]
        num_read=0
        for f_i in hdf5_files:
            if '_'+algs[i]+'_' in f_i:
                try:
                    f=df.HDF5File(bip.pde.mpi_comm,os.path.join(fld_m,f_i),"r")
                    # samp_mean.zero(); #samp_std.zero(); #num_read=0
                    err_=0
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
                        # samp_mean.axpy(wts[num_read][s],u)
                        # samp_std.axpy(wts[num_read][s],u*u)
                        err_+=wts[num_read][s]*(u-true_param).norm('l2')**2
                    # compute error
                    # errs.append((samp_mean-true_param).norm('l2')/true_param.norm('l2'))
                    errs.append(np.sqrt(err_)/true_param.norm('l2'))
                    num_read+=1
                    f.close()
                    print(f_i+' has been read!')
                    f_read=f_i
                except:
                    pass
        print('%d experiment(s) have been processed for %s algorithm with %s likelihood model.' % (num_read, algs[i], lik_mdls[m]))
        if num_read>0:
#             samp_mean=samp_mean/num_read; samp_std=samp_std/num_read
            # mean_v[i].set_local(samp_mean)
            # std_v[i].set_local(np.sqrt((samp_std - samp_mean*samp_mean).get_local()))
            errs = np.stack(errs)
            mse_m[m,i] = np.median(errs)
            mse_s[m,i] = errs.std()
# save
import pandas as pd
mse_m = pd.DataFrame(data=mse_m,index=lik_mdls,columns=alg_names[:num_algs])
mse_s = pd.DataFrame(data=mse_s,index=lik_mdls,columns=alg_names[:num_algs])
mse_m.to_csv(os.path.join(folder,'MSE-mean.csv'),columns=alg_names[:num_algs])
mse_s.to_csv(os.path.join(folder,'MSE-std.csv'),columns=alg_names[:num_algs])