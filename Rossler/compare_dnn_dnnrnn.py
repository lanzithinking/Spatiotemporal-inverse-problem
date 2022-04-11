"""
This is to compare the emulation effect by DNN and DNN-RNN(LSTM).
"""

import numpy as np
import tensorflow as tf
import sys,os
sys.path.append( '../' )
from Rossler import Rossler
from nn.dnn import DNN
from nn.dnn_rnn import DNN_RNN
from tensorflow.keras.models import load_model
import timeit,pickle
import matplotlib.pyplot as plt

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# likelihood models
mdls=('simple','STlik')
n_mdl=len(mdls)
emus=['DNN','DNN_RNN']
# set random seed
seeds = [2021+i*10 for i in range(10)]
n_seed = len(seeds)
# training/testing sizes
train_sizes = [100,500,1000,5000,10000]
n_train = len(train_sizes)
test_size = 100
# save relative errors and times
fun_errors = np.zeros((n_mdl,n_seed,n_train,test_size))
grad_errors = np.zeros((n_mdl,n_seed,n_train,test_size))
train_times = np.zeros((n_mdl,n_seed,n_train))
test_times = np.zeros((2*n_mdl,n_seed,n_train))

## define Rossler inverse problem ##
num_traj = 1
t_init = 1000
t_final = 1100
time_res = 100
obs_times = np.linspace(t_init, t_final, time_res)
var_out = True # True; 'cov'; False
# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1
ensbl_sz = 500

# data folder
folder = './train_NN/'

try:
    with open(os.path.join(folder,'compare_dnn_dnnrnn.pckl'),'rb') as f:
        fun_errors,grad_errors,train_times,test_times=pickle.load(f)
    print('Comparison results loaded!')
except:
    print('Obtaining comparison results...\n')
    for m in range(n_mdl):
        print('Working on '+mdls[m]+' model...\n')
        # define inverse problem
        avg_traj = {'simple':'aug','STlik':False}[mdls[m]] # True; 'aug'; False
        STlik = (mdls[m]=='STlik')
        rsl = Rossler(num_traj=num_traj, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, STlik=STlik)
        input_dim=len(rsl.ode_params)
        # load training data
        loaded=np.load(file=os.path.join(folder,mdls[m]+'_'+algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
        X=loaded['X']
        Y=loaded['Y']
        num_samp=X.shape[0]
        loglik = lambda y: -rsl.misfit.cost(obs=y)
        # define emulators
        if emus[m]=='DNN':
            output_dim=rsl.misfit.obs.size
            depth=4
            node_sizes=[input_dim,30,100,output_dim]
            activations={'hidden':tf.keras.layers.LeakyReLU(alpha=0.01),'output':'linear'}
            droprate=0.0
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005,amsgrad=True)
            emulator=DNN(X.shape[1], Y.shape[1], depth=depth, node_sizes=node_sizes, droprate=droprate,
                         activations=activations, optimizer=optimizer)
            # loglik = lambda y: -0.5*tf.math.reduce_sum((y-rsl.misfit.obs)**2/rsl.misfit.nzvar,axis=1)
        elif emus[m]=='DNN_RNN':
            temp_dim,spat_dim=rsl.misfit.obs.shape[1:]
            Y=Y.reshape((-1,temp_dim,spat_dim))
            output_dim = np.prod(rsl.misfit.obs.shape[1:])
            depth=4
            node_sizes=[input_dim,30,100,output_dim]
            activations={'hidden':'softplus','output':'sigmoid','gru':'linear'}
            droprate=.5
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005,amsgrad=True)
            W = tf.convert_to_tensor(rsl.misfit.stgp.tomat(),dtype=tf.float32)
            # loglik = lambda y: -0.5*tf.math.reduce_sum(tf.reshape(y-rsl.misfit.obs,[-1,output_dim])*tf.transpose(tf.linalg.solve(W,tf.transpose(tf.reshape(y-rsl.misfit.obs,[-1,output_dim])))),axis=1)
            custom_loss = lambda y_true, y_pred: [tf.abs(loglik(y_true)-loglik(y_pred)), tf.linalg.solve(W[None,:,:],tf.reshape(y_true-y_pred,[-1,output_dim,1]))]
            emulator=DNN_RNN(X.shape[1], Y.shape[1:], depth=depth, node_sizes=node_sizes, droprate=droprate,
                             activations=activations, optimizer=optimizer, loss=custom_loss)
        for s in range(n_seed):
            np.random.seed(seeds[s])
            tf.random.set_seed(seeds[s])
            prng=np.random.RandomState(seeds[s])
            for t in range(n_train):
                # select training and testing data
                sel4train = prng.choice(num_samp,size=train_sizes[t],replace=False)
                sel4test = prng.choice(num_samp,size=test_size,replace=False) if train_sizes[t]+test_size>num_samp else prng.choice(np.setdiff1d(range(num_samp),sel4train),size=test_size,replace=False)
                
                # train emulator
                f_name=emus[m]+'_'+algs[alg_no]+'_'+str(ensbl_sz)+'_seed'+str(seeds[s])+'_trainsz'+str(train_sizes[t])
                try:
                    emulator.model=load_model(os.path.join(folder+emus[m],f_name+'.h5'),custom_objects={'loss':None})
                    print(f_name+' has been loaded!')
                except:
                    try:
                        emulator.model.load_weights(os.path.join(folder+emus[m],f_name+'.h5'))
                        print(f_name+' has been loaded!')
                    except Exception as err:
                        print(err)
                        print('Train '+emus[m]+' with seed {} and training size {}...\n'.format(seeds[s],train_sizes[t]))
                        epochs=500
                        patience=5
                        t_start=timeit.default_timer()
                        try:
                            emulator.train(X[sel4train],Y[sel4train],x_test=X[sel4test],y_test=Y[sel4test],epochs=epochs,batch_size=64,verbose=1,patience=patience)
                        except Exception as err:
                            print(err)
                            pass
                        t_used=timeit.default_timer()-t_start
                        train_times[m,s,t]=t_used
                        print('\nTime used for training '+emus[m]+': {}'.format(t_used))
                        # save CNN
                        save_dir=folder+emus[m]
                        if not os.path.exists(save_dir): os.makedirs(save_dir)
                        try:
                            emulator.model.save(os.path.join(save_dir,f_name+'.h5'))
                        except:
                            emulator.model.save_weights(os.path.join(save_dir,f_name+'.h5'))
                
                # test
                logLik = lambda x: loglik(emulator.model(x))
                t_used = np.zeros(2)
                print('Testing trained models...\n')
                for n in range(test_size):
                    u=X[sel4test][n]
                    # exact calculation 
                    t_start=timeit.default_timer()
                    # if emus[m]=='DNN':
                    #     ll_xact,dll_xact = rsl.get_geom(u,[0,1])[:2]
                    # elif emus[m]=='DNN_RNN':
                    #     ll_xact = -rsl._get_misfit(u,option='quad')
                    #     dll_xact = -rsl._get_grad(u)
                    ll_xact = rsl.get_geom(u)[0]
                    t_used[0] += timeit.default_timer()-t_start
                    
                    # emulation 
                    t_start=timeit.default_timer()
                    ll_emul = logLik(u[None,:])#.numpy()[0]
                    # dll_emul = emulator.gradient(u[None,:], logLik)
                    t_used[1] += timeit.default_timer()-t_start
                    
                    # record difference
                    dif_fun = np.abs(ll_xact - ll_emul)
                    # dif_grad = dll_xact - dll_emul
                    fun_errors[m,s,t,n]=dif_fun/abs(ll_xact)
                    # grad_errors[m,s,t,n]=np.linalg.norm(dif_grad)/np.linalg.norm(dll_xact)
                    
                print('Time used for calculation: {}'.format(t_used[0])+' vs '+emus[m]+'-emulation: {}'.format(t_used[1]))
                test_times[m,s,t]=t_used[1]; test_times[2+m,s,t]=t_used[0]
    
    # save results
    with open(os.path.join(folder,'compare_dnn_dnnrnn.pckl'),'wb') as f:
        pickle.dump([fun_errors,grad_errors,train_times,test_times],f)

# make some pots
import pandas as pd
import seaborn as sns
# prepare for the data
alg_array=np.hstack((['DNN']*n_seed*n_train,['DNN_RNN']*n_seed*n_train))
trs_array=np.zeros((2,n_seed,n_train),dtype=np.int)
for t in range(n_train): trs_array[:,:,t]=train_sizes[t]
fun_err_array=np.median(fun_errors,axis=3)
grad_err_array=np.median(grad_errors,axis=3)
train_time_array=train_times
df_err=pd.DataFrame({'algorithm':alg_array.flatten(),
                     'training_size':trs_array.flatten(),
                     'function_error':fun_err_array.flatten(),
                     'gradient_error':grad_err_array.flatten(),
                     'training_time':train_time_array.flatten(),
#                      'testing_time':test_time_array.flatten()
                     })

alg_array=np.hstack((['DNN']*n_seed*n_train,['DNN_RNN']*n_seed*n_train,['time_avgerate']*n_seed*n_train,['STGP']*n_seed*n_train))
trs_array=np.zeros((2*n_mdl,n_seed,n_train),dtype=np.int)
for t in range(n_train): trs_array[:,:,t]=train_sizes[t]
test_time_array=test_times
df_time=pd.DataFrame({'algorithm':alg_array.flatten(),
                      'training_size':trs_array.flatten(),
                      'testing_time':test_time_array.flatten()
                     })

# plot errors
sns.set(font_scale=1.2)
# fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
fig,axes = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=False,figsize=(7,5))
# plot
# plt.axes(axes.flat[0])
sns.barplot(x='training_size',y='function_error',hue='algorithm',data=df_err,errwidth=1,capsize=.1)
# plt.title('Error of Function')
plt.gca().legend().set_title('')
# plt.axes(axes.flat[1])
# sns.barplot(x='training_size',y='gradient_error',hue='algorithm',data=df_err,errwidth=1,capsize=.1)
# plt.title('Error of Gradient')
# plt.ylim(.5,1.5)
plt.gca().legend().set_title('')
# save plots
# fig.tight_layout(h_pad=1)
# plt.savefig(os.path.join(folder,'error_dnn_dnnrnn.png'),bbox_inches='tight')
plt.savefig(os.path.join(folder,'relerr_dnn_dnnrnn.png'),bbox_inches='tight')
# plt.show()

# plot times
fig,axes = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=False,figsize=(12,5))
# plot
plt.axes(axes.flat[0])
# sns.pointplot(x='training_size',y='training_time',hue='algorithm',data=df_err,errwidth=.8,capsize=.1,scale=.5)
sns.pointplot(x='training_size',y='training_time',hue='algorithm',data=df_err,ci=None)
# plt.title('Training Time')
plt.gca().legend().set_title('')
plt.axes(axes.flat[1])
sns.pointplot(x='training_size',y='testing_time',hue='algorithm',data=df_time,ci=None)
# plt.title('Testint Time')
plt.gca().legend().set_title('')
# save plots
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'time_dnn_dnnrnn.png'),bbox_inches='tight')
# plt.show()