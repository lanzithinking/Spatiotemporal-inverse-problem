"""
This is to train DNN for simple model and to test its emulation effect.
"""

import numpy as np
import tensorflow as tf
import sys,os,pickle
sys.path.append( '../' )
from Chen import Chen
from nn.dnn import DNN
from tensorflow.keras.models import load_model

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
seed=2021
np.random.seed(seed)
tf.random.set_seed(seed)

# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1
mdls=('simple','STlik')
num_mdls=len(mdls)
mdl_no=0 # DNN for simple likelihood model
## define Chen inverse problem ##
num_traj = 1 # only consider single trajectory!
prior_params = {'mean':[3.5, 1.2, 3.3], 'std':[0.35, 0.5, 0.15]}
t_init = 100
t_final = 110
time_res = 100
obs_times = np.linspace(t_init, t_final, time_res)
avg_traj = {'simple':'aug','STlik':False}[mdls[mdl_no]] # True; 'aug'; False
var_out = 'cov' # True; 'cov'; False
STlik = (mdls[mdl_no]=='STlik')
chn = Chen(num_traj=num_traj, prior_params=prior_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik) # set STlik=False for simple likelihood; STlik has to be used with avg_traj

# define the emulator (DNN)
# load data
ensbl_sz = 500
folder = './train_NN'
loaded=np.load(file=os.path.join(folder,mdls[mdl_no]+'_'+algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
X=loaded['X']
Y=loaded['Y']
# pre-processing: scale X to 0-1
# X-=np.nanmin(X,axis=(1,2),keepdims=True) # try axis=(1,2,3)
# X/=np.nanmax(X,axis=(1,2),keepdims=True)
# split train/test
num_samp=X.shape[0]
# n_tr=np.int(num_samp*.75)
# x_train,y_train=X[:n_tr],Y[:n_tr]
# x_test,y_test=X[n_tr:],Y[n_tr:]
tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
x_train,x_test=X[tr_idx],X[te_idx]
y_train,y_test=Y[tr_idx],Y[te_idx]

# define DNN
input_dim=len(chn.ode_params)
output_dim=chn.misfit.obs.size
depth=4
node_sizes=[input_dim,30,100,output_dim]
# activations={'hidden':tf.keras.layers.LeakyReLU(alpha=0.01),'output':'linear'}
# activations={'hidden':tf.math.sin,'output':'linear'}
activations={'hidden':'softplus','output':'linear'}
droprate=0.0
# sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
kernel_initializers={'hidden':'he_uniform','output':'he_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,amsgrad=True)
# optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001)
# dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, node_sizes=node_sizes, droprate=droprate,
#         activations=activations, optimizer=optimizer)
# loglik = lambda y: -chn.misfit.cost(obs=y)
W = tf.convert_to_tensor(chn.misfit.nzvar[0],dtype=tf.float32) # chn.misfit.nzvar[0]
# loglik = lambda y: -0.5*tf.math.reduce_sum((y-chn.misfit.obs)*tf.linalg.solve(W,(y-chn.misfit.obs)[:,:,None])[:,:,0],axis=1)
loglik = lambda y: -0.5*tf.math.reduce_sum((y-chn.misfit.obs)*tf.transpose(tf.linalg.solve(W,tf.transpose(y-chn.misfit.obs))),axis=1)
custom_loss = lambda y_true, y_pred: [tf.abs(loglik(y_true)-loglik(y_pred)), tf.linalg.solve(W,y_true-y_pred)]
# custom_loss = lambda y_true, y_pred: [tf.math.reduce_sum((y_true-y_pred)*tf.transpose(tf.linalg.solve(W,tf.transpose(y_true-y_pred))),axis=1), None]
dnn=DNN(x_train.shape[1], y_train.shape[1], depth=depth, node_sizes=node_sizes, droprate=droprate,
        activations=activations, optimizer=optimizer, loss=custom_loss)
savepath=folder+'/DNN/saved_model'
if not os.path.exists(savepath): os.makedirs(savepath)
import time
ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
f_name='dnn_'+algs[alg_no]+'_'+str(ensbl_sz)+'-'+ctime
try:
#     dnn.model=load_model(os.path.join(savepath,f_name+'.h5'))
#     dnn.model=load_model(os.path.join(savepath,f_name+'.h5'),custom_objects={'loss':None})
    dnn.model.load_weights(os.path.join(savepath,f_name+'.h5'))
    print(f_name+' has been loaded!')
except Exception as err:
    print(err)
    print('Train DNN...\n')
    epochs=500
    patience=5
    import timeit
    t_start=timeit.default_timer()
    dnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
    t_used=timeit.default_timer()-t_start
    print('\nTime used for training DNN: {}'.format(t_used))
    # save DNN
#     dnn.model.save(os.path.join(savepath,f_name+'.h5'))
#     dnn.save(savepath,f_name) # fails due to the custom kernel_initializer
    dnn.model.save_weights(os.path.join(savepath,f_name+'.h5'))

# select some gradients to evaluate and compare
logLik = lambda x: loglik(dnn.model(x))
import timeit
t_used = np.zeros(2)
n_dif = 1000
dif = np.zeros((n_dif,2))
loaded=np.load(file=os.path.join(folder,mdls[mdl_no]+'_'+algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY'+'.npz'))
prng=np.random.RandomState(2021)
sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
X=loaded['X'][sel4eval]; Y=loaded['Y'][sel4eval]
sel4print = prng.choice(n_dif,size=10,replace=False)
prog=np.ceil(n_dif*(.1+np.arange(0,1,.1)))
for n in range(n_dif):
    u=X[n]
    # calculate gradient
    t_start=timeit.default_timer()
    ll_xact,dll_xact = chn.get_geom(u,[0,1])[:2]
    t_used[0] += timeit.default_timer()-t_start
    # emulate gradient
    t_start=timeit.default_timer()
    ll_emul = logLik(u[None,:]).numpy()
    dll_emul = dnn.gradient(u[None,:], logLik)
    t_used[1] += timeit.default_timer()-t_start
    # test difference
    dif_fun = np.abs(ll_xact - ll_emul)
    dif_grad = dll_xact - dll_emul
    dif[n] = np.array([dif_fun, np.linalg.norm(dif_grad)/np.linalg.norm(dll_xact)])

#     # check the gradient extracted from emulation
#     v=chn.sample()
#     h=1e-4
#     dll_emul_fd_v=(logLik(u[None,:]+h*v[None,:])-logLik(u[None,:]))/h
#     reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v))/np.linalg.norm(v)
#     print('Relative difference between finite difference and extracted results: {}'.format(reldif))

    if n+1 in prog:
        print('{0:.0f}% evaluation has been completed.'.format(np.float(n+1)/n_dif*100))

    # plot
    if n in sel4print:
        print('Difference between the calculated and emulated values: {}'.format(dif_fun))
        print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})\n'.format(dif_grad.min(),np.median(dif_grad),dif_grad.max()))

print('Time used to calculate vs emulate gradients: {} vs {}'.format(*t_used.tolist()))
# save to file
import pandas as pd
savepath='./train_NN/DNN/summary'
if not os.path.exists(savepath): os.makedirs(savepath)
file=os.path.join(savepath,'dif-'+ctime+'.txt')
np.savetxt(file,dif)
con_str=np.array2string(np.array(node_sizes),separator=',').replace('[','').replace(']','') if 'node_sizes' in locals() or 'node_sizes' in globals() else str(depth)
act_str=','.join([val.__name__ if type(val).__name__=='function' else val.name if callable(val) else val for val in activations.values()])
dif_fun_sumry=[dif[:,0].min(),np.median(dif[:,0]),dif[:,0].max()]
dif_fun_str=np.array2string(np.array(dif_fun_sumry),precision=2,separator=',').replace('[','').replace(']','') # formatter={'float': '{: 0.2e}'.format}
dif_grad_sumry=[dif[:,1].min(),np.median(dif[:,1]),dif[:,1].max()]
dif_grad_str=np.array2string(np.array(dif_grad_sumry),precision=2,separator=',').replace('[','').replace(']','')
sumry_header=('Time','depth/node_sizes','activations','droprate','dif_fun (min,med,max)','dif_grad (min,med,max)')
sumry_np=np.array([ctime,con_str,act_str,droprate,dif_fun_str,dif_grad_str])
file=os.path.join(savepath,'dif_sumry.txt')
if not os.path.isfile(file):
    np.savetxt(file,sumry_np[None,:],fmt="%s",delimiter='\t|',header='\t|'.join(sumry_header))
else:
    with open(file, "ab") as f:
        np.savetxt(f,sumry_np[None,:],fmt="%s",delimiter='\t|')
sumry_pd=pd.DataFrame(data=[sumry_np],columns=sumry_header)
file=os.path.join(savepath,'dif_sumry.csv')
if not os.path.isfile(file):
    sumry_pd.to_csv(file,index=False,header=sumry_header)
else:
    sumry_pd.to_csv(file,index=False,mode='a',header=False)


# plot
import matplotlib.pyplot as plt
# from joblib import Parallel, delayed
# import multiprocessing
# n_jobs = np.min([10, multiprocessing.cpu_count()])
# plt.rcParams['image.cmap'] = 'jet'
# fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
# sub_figs=[None]*2
#
# true_input=list(chn.misfit.true_params.values())
# dim=[0,1]
# levels=20
# grad=True
#
# for i,ax in enumerate(axes):
#     x=np.linspace(true_input[dim[0]]-1.,true_input[dim[0]]+1.)
#     y=np.linspace(true_input[dim[1]]-.2,true_input[dim[1]]+.2)
#     X,Y=np.meshgrid(x,y)
#     Input=np.zeros((X.size,input_dim))
#     Input[:,dim[0]],Input[:,dim[1]]=X.flatten(),Y.flatten()
#     # Z=np.reshape([chn.get_geom(u)[0] for u in Input], X.shape) if i==0 else logLik(Input).numpy().reshape(X.shape)
#     Z=np.reshape(Parallel(n_jobs=n_jobs)(delayed(chn.get_geom)(u)[0] for u in Input), X.shape) if i==0 else logLik(Input).numpy().reshape(X.shape)
#     if grad:
#         x=np.linspace(true_input[dim[0]]-1.,true_input[dim[0]]+1.,10)
#         y=np.linspace(true_input[dim[1]]-.2,true_input[dim[1]]+.2,10)
#         X_,Y_=np.meshgrid(x,y)
#         Input=np.zeros((X_.size,input_dim))
#         Input[:,dim[0]],Input[:,dim[1]]=X_.flatten(),Y_.flatten()
#         # G=np.array([chn.get_geom(u,geom_ord=[0,1])[1] for u in Input]) if i==0 else dnn.gradient(Input, logLik)
#         G=np.array(Parallel(n_jobs=n_jobs)(delayed(chn.get_geom)(u,geom_ord=[0,1])[1] for u in Input)) if i==0 else dnn.gradient(Input, logLik)
#         U,V=G[:,dim[0]].reshape(X_.shape),G[:,dim[1]].reshape(X_.shape)
#     sub_figs[i]=ax.contourf(X,Y,Z,levels)
#     axes.flat[i].set_xlabel('$u_{}$'.format(dim[0]+1))
#     axes.flat[i].set_ylabel('$u_{}$'.format(dim[1]+1),rotation=0)
#     if grad: ax.quiver(X_,Y_,U,V)
#     plt.title(('Calculated','Emulated')[i])
#
# from util.common_colorbar import common_colorbar
# fig=common_colorbar(fig,axes,sub_figs)
# plt.subplots_adjust(wspace=0.2, hspace=0)
# # save plots
# # fig.tight_layout(h_pad=1)
# plt.savefig(os.path.join(savepath,f_name+'.png'),bbox_inches='tight')
# # plt.show()


import pandas as pd
import seaborn as sns
# define marginal density plot
def plot_pdf(x, **kwargs):
    nx = len(x)
    # z = np.zeros(nx)
    para0 = kwargs.pop('para0',None)
    f = kwargs.pop('f',None)
    # for i in range(nx):
    #     para_ = para0.copy()
    #     para_[x.name] = x[i]
    #     z[i] = f(np.array(list(para_.values()))[None,:])
    params=np.tile(list(para0.values()),(nx,1))
    params[:,list(para0.keys()).index(x.name)]=x
    z=f(params)
    
    plt.plot(x, z, **kwargs)

# define contour function
def contour(x, y, **kwargs):
    nx = len(x); ny = len(y)
    # z = np.zeros((nx, ny))
    para0 = kwargs.pop('para0',None)
    f = kwargs.pop('f',None)
    # for i in range(nx):
    #     for j in range(ny):
    #         para_ = para0.copy()
    #         para_[x.name] = x[i]; para_[y.name] = y[j]
    #         z[i,j] = f(np.array(list(para_.values()))[None,:])
    params=np.tile(list(para0.values()),(nx*ny,1))
    params[:,list(para0.keys()).index(x.name)]=np.tile(x,ny)
    params[:,list(para0.keys()).index(y.name)]=np.repeat(y,nx)
    z=np.reshape(f(params).numpy(),(nx,ny),order='F')
    
    plt.contourf(x, y, z, levels=np.quantile(z,[.67,.9,.99]), **kwargs)

# prepare for plotting data
para0 = chn.misfit.true_params
marg = [1,.2,1]; res = 100
grid_data = chn.misfit.true_params.copy()
for i,k in enumerate(grid_data):
    grid_data[k] = np.linspace(grid_data[k]-marg[i],grid_data[k]+marg[i], num=res)
grid_data = pd.DataFrame(grid_data)
# plot
sns.set(font_scale=1.1)
import time
t_start=time.time()
g = sns.PairGrid(grid_data, diag_sharey=False, corner=True, size=3)
g.map_diag(plot_pdf, para0=para0, f=lambda param:-logLik(param))
# g.map_lower(contour, para0=para0, f=lambda param:np.exp(logLik(param)), cmap='gray')
g.map_lower(contour, para0=para0, f=lambda param:logLik(param), cmap='gray')
# for ax in g.axes.flatten():
#     # rotate x axis labels
#     # ax.set_xlabel(ax.get_xlabel(), rotation = 90)
#     # rotate y axis labels
#     ax.set_ylabel(ax.get_ylabel(), rotation = 0)
#     # set y labels alignment
#     ax.yaxis.get_label().set_horizontalalignment('right')
g.savefig(os.path.join(savepath,f_name+'.png'),bbox_inches='tight')
t_end=time.time()
print('time used: %.5f'% (t_end-t_start))