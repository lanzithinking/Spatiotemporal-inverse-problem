"""
This is to compare simple and STlik in emulating (extracted) loglik taking reference to those exactly calculated.
"""

import numpy as np
import tensorflow as tf
import sys,os
from lorenz import Lorenz

sys.path.append( '../' )

from nn.dnn import DNN
from nn.dnn_rnn import DNN_RNN
from tensorflow.keras.models import load_model
import timeit,pickle
import matplotlib.pyplot as plt

# tf.compat.v1.disable_eager_execution() # needed to train with custom loss # comment to plot
# set random seed
seed = 2021
seeds = [2020+i*10 for i in range(10)]
n_seed = len(seeds)
# training/testing sizes
train_sizes = [50,100,200,500,1000]
n_train = len(train_sizes)
test_size = 100
# save relative errors and times
fun_errors = np.zeros((2,n_seed,n_train,test_size))
remfun_errors = np.zeros((2,n_seed,n_train,test_size))
train_times = np.zeros((2,n_seed,n_train))
pred_times = np.zeros((4,n_seed,n_train))

# define the inverse problem
num_traj = 1 # only consider single trajectory!
prior_params = {'mean':[2.0, 1.2, 3.3], 'std':[0.2, 0.5, 0.15]}
t_init = 100
t_final = 110
time_res = 100
obs_times = np.linspace(t_init, t_final, time_res)
avg_traj = {'simple':'aug','STlik':False}['simple'] # True; 'aug'; False
var_out = 'cov' # True; 'cov'; False
STlik = False
lrz_sim = Lorenz(num_traj=num_traj, prior_params=prior_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik) # set STlik=False for simple likelihood; STlik has to be used with avg_traj

avg_traj = {'simple':'aug','STlik':False}['STlik'] # True; 'aug'; False
var_out = 'cov' # True; 'cov'; False
STlik = True
lrz_stlik = Lorenz(num_traj=num_traj, prior_params=prior_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik)
temp_dim,spat_dim=lrz_stlik.misfit.obs[0].shape

# algorithms
algs=['EKI','EKS']
num_algs=len(algs)
alg_no=1
ensbl_sz = 500

# load data
folder = './train_NN'

# load data for DNN
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
Xdnn=loaded['X']
Ydnn=loaded['Y']

# define DNN
depth=2
# node_sizes=[X.shape[1],4096,2048,1024,Y.shape[1]]
#activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear','lstm':'tanh'}
activations={'hidden':tf.keras.layers.LeakyReLU(alpha=.01),'output':'linear'}
droprate=0.2
#kernel_initializers={'hidden':sin_init,'output':'glorot_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)###1,amsgrad=True?????????????????####

dnn=DNN(Xdnn.shape[1], Ydnn.shape[1], depth=depth, droprate=droprate, activations=activations, optimizer=optimizer)


# load data for DNN-RNN
loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY_stlik.npz'))
Xr=loaded['X']
Yr=loaded['Y']
Yr=Yr.reshape((-1,temp_dim,spat_dim))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# transform data
Xr = scaler.fit_transform(Xr)

# define DNN-RNN
depth=2
#activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear','lstm':'tanh'}
activations={'hidden':'relu','latent':'linear','output':'linear','lstm':'tanh'}
#activations={'conv':tf.math.sin,'latent':tf.math.sin,'output':'linear','lstm':'tanh'}
latent_dim=Yr.shape[2]
droprate=0.2
#kernel_initializers={'conv':sin_init,'latent':sin_init,'output':'glorot_uniform'}
kernel_initializers={'hidden':'he_uniform','latent':'glorot_uniform','output':'glorot_uniform'}
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005,amsgrad=True,clipnorm=1)###clipnorm=1.?????????????????####

dnnrnn=DNN_RNN(Xr.shape[1], Yr.shape[1:], depth=depth, latent_dim=latent_dim, droprate=droprate,
               activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer)


# split train/test
num_samp=Xdnn.shape[0]
folder = './train_NN'
if not os.path.exists(folder): os.makedirs(folder)

try:
    with open(os.path.join(folder,'compare_sim_stlik.pckl'),'rb') as f:
        fun_errors,remfun_errors,train_times,pred_times=pickle.load(f)
    print('Comparison results loaded!')
except:
    print('Obtaining comparison results...\n')
    for s in range(n_seed):
        np.random.seed(seeds[s])
        tf.random.set_seed(seeds[s])
        prng=np.random.RandomState(seeds[s])
        for t in range(n_train):
            # select training and testing data
            sel4train = prng.choice(num_samp,size=train_sizes[t],replace=False)
            sel4test = prng.choice(np.setdiff1d(range(num_samp),sel4train),size=test_size,replace=False)
            
            # train dnn
            f_name='dnn_'+algs[alg_no]+str(ensbl_sz)+'_seed'+str(seeds[s])+'_trainsz'+str(train_sizes[t])
            try:
                dnn.model=load_model(os.path.join(folder+'/DNN/',f_name+'.h5'))
                print(f_name+' has been loaded!')
            except Exception as err:
                print(err)
                print('Train DNN model with seed {} and training size {}...\n'.format(seeds[s],train_sizes[t]))
                epochs=200
                patience=0
                t_start=timeit.default_timer()
                try:
                    dnn.train(Xdnn[sel4train],Ydnn[sel4train],x_test=Xdnn[sel4test],y_test=Ydnn[sel4test],epochs=epochs,batch_size=64,verbose=1,patience=patience)
                except Exception as err:
                    print(err)
                    pass
                t_used=timeit.default_timer()-t_start
                train_times[0,s,t]=t_used
                print('\nTime used for training DNN: {}'.format(t_used))
                # save DNN
                save_dir=folder+'/DNN/'
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                try:
                    dnn.model.save(os.path.join(save_dir,f_name+'.h5'))
                except:
                    dnn.model.save_weights(os.path.join(save_dir,f_name+'.h5'))
        
            # train DNN-RNN
            f_name='dnnrnn_'+algs[alg_no]+str(ensbl_sz)+'_seed'+str(seeds[s])+'_trainsz'+str(train_sizes[t])
            try:
                dnnrnn.model=load_model(os.path.join(folder+'/DNN_RNN/',f_name+'.h5'))
                print(f_name+' has been loaded!')
            except:
                try:
                    dnnrnn.model.load_weights(os.path.join(folder+'/DNN_RNN/',f_name+'.h5'))
                    print(f_name+' has been loaded!')
                except Exception as err:
                    print(err)
                    print('Train DNNRNN with seed {} and training size {}...\n'.format(seeds[s],train_sizes[t]))
                    epochs=200
                    patience=0
                    t_start=timeit.default_timer()
                    try:
                        dnnrnn.train(Xr[sel4train],Yr[sel4train],x_test=Xr[sel4test],y_test=Yr[sel4test],epochs=epochs,batch_size=64,verbose=1,patience=patience)
                    except Exception as err:
                        print(err)
                        pass
                    t_used=timeit.default_timer()-t_start
                    train_times[1,s,t]=t_used
                    print('\nTime used for training DNNRNN: {}'.format(t_used))
                    # save DNNRNN
                    save_dir=folder+'/DNN_RNN/'
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    try:
                        dnnrnn.model.save(os.path.join(save_dir,f_name+'.h5'))
                    except:
                        dnnrnn.model.save_weights(os.path.join(save_dir,f_name+'.h5'))
            
            # test
            loglik_sim = lambda x: -lrz_sim.misfit.cost(obs=dnn.model(x))
            loglik_stlik = lambda x: -lrz_stlik.misfit.cost(obs=dnnrnn.model(x))

            t_used = np.zeros(4)
            print('Testing trained models...\n')
            for n in range(test_size):
                u=(Xdnn[sel4test][n])
                # calculate truth
                t_start=timeit.default_timer()
                ll_xact,dll_xact = lrz_sim.get_geom(u,[0,1])[:2]
                t_used[0] += timeit.default_timer()-t_start
                
                # emulation by DNN
                t_start=timeit.default_timer()
                ll_emul = loglik_sim(u[None,:])
                t_used[1] += timeit.default_timer()-t_start
                # record difference
                dif_fun = np.abs(ll_xact - ll_emul)
                fun_errors[0,s,t,n]=dif_fun
                remfun_errors[0,s,t,n]=np.linalg.norm(dif_fun)/np.linalg.norm(ll_xact)
                
                u=(Xr[sel4test][n])
                # calculate truth for stlik
                t_start=timeit.default_timer()
                ll_xact,dll_xact = lrz_stlik.get_geom(u,[0,1])[:2]
                t_used[2] += timeit.default_timer()-t_start
                # emulation by dnnrnn
                t_start=timeit.default_timer()
                ll_emul = loglik_stlik(u[None,:])
                t_used[3] += timeit.default_timer()-t_start
                # record difference
                dif_fun = np.abs(ll_xact - ll_emul)
                fun_errors[1,s,t,n]=dif_fun
                remfun_errors[1,s,t,n]=np.linalg.norm(dif_fun)/np.linalg.norm(ll_xact)
                
            print('Time used for calculation simple: {} vs DNN-emulation: {} vs calculation stlik: {} vs DNNRNN-emulation: {}'.format(*t_used.tolist()))
            # dnn, dnnrnn, calculate simple, calculate stlik
            pred_times[0,s,t]=t_used[1]; pred_times[1,s,t]=t_used[3]; pred_times[2,s,t]=t_used[0]; pred_times[3,s,t]=t_used[2]
    
    # save results
    with open(os.path.join(folder,'compare_sim_stlik.pckl'),'wb') as f:
        pickle.dump([fun_errors,remfun_errors,train_times,pred_times],f)

# make some pots
import pandas as pd
import seaborn as sns
# prepare for the data
alg_array=np.hstack((['DNN']*n_seed*n_train,['DNNRNN']*n_seed*n_train))
trs_array=np.zeros((2,n_seed,n_train),dtype=np.int)
for t in range(n_train): trs_array[:,:,t]=train_sizes[t]
fun_err_array=np.median(fun_errors,axis=3)
remfun_err_array=np.median(remfun_errors,axis=3)

train_time_array=train_times
# test_time_array=pred_times[:2]
df_err=pd.DataFrame({'algorithm':alg_array.flatten(),
                     'training_size':trs_array.flatten(),
                     'function_error':fun_err_array.flatten(),
                     'remfunction_error':remfun_err_array.flatten(),
                     'training_time':train_time_array.flatten(),
#                      'testing_time':test_time_array.flatten()
                     })

alg_array=np.hstack((['DNN']*n_seed*n_train,['DNNRNN']*n_seed*n_train,['FE_sim']*n_seed*n_train,['FE_stlik']*n_seed*n_train))
trs_array=np.zeros((4,n_seed,n_train),dtype=np.int)
for t in range(n_train): trs_array[:,:,t]=train_sizes[t]
test_time_array=pred_times
df_time=pd.DataFrame({'algorithm':alg_array.flatten(),
                      'training_size':trs_array.flatten(),
                      'testing_time':test_time_array.flatten()
                     })
    
    
df_err.to_csv(os.path.join(folder,'error_sim_stlik_sumry.csv'),index=False)
df_time.to_csv(os.path.join(folder,'time_sim_stlik_sumry.csv'),index=False)

# plot errors
fig,axes = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=False,figsize=(12,5))
# plot
plt.axes(axes)
sns.barplot(x='training_size',y='function_error',hue='algorithm',data=df_err,errwidth=1,capsize=.1)
# plt.title('Error of Function')
plt.gca().legend().set_title('')
#plt.ylim(.5,1.5)
# fig.tight_layout(h_pad=1)
plt.savefig(os.path.join(folder,'error_sim_stlik.png'),bbox_inches='tight')
# plt.show()

# plot relative errors
fig,axes = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=False,figsize=(12,5))
# plot
plt.axes(axes)
sns.barplot(x='training_size',y='remfunction_error',hue='algorithm',data=df_err,errwidth=1,capsize=.1)
plt.gca().legend().set_title('')
plt.savefig(os.path.join(folder,'rem_sim_stlik.png'),bbox_inches='tight')

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
plt.savefig(os.path.join(folder,'time_sim_stlik.png'),bbox_inches='tight')