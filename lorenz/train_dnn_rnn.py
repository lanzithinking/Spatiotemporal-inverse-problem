"""
This is to test DNN-RNN in emulating (extracted) gradients compared with those exactly calculated.
"""

import numpy as np
import tensorflow as tf
import sys,os,argparse
from lorenz import Lorenz
sys.path.append( '../' )
from nn.dnn_rnn import DNN_RNN
from tensorflow.keras.models import load_model

# tf.compat.v1.disable_eager_execution() 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience', nargs='?', type=int, default=0)
    parser.add_argument('--depth', nargs='?', type=int, default=2)
    parser.add_argument('--droprate', type=float, default=.2)
    parser.add_argument('--lrrate', type=float, default=.0005)
    
    args = parser.parse_args()
    # set random seed
    seed=2021
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    ## define Lorenz inverse problem ##
    num_traj = 1 # only consider single trajectory!
    prior_params = {'mean':[2.0, 1.2, 3.3], 'std':[0.2, 0.5, 0.15]}
    t_init = 100
    t_final = 110
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = {'simple':'aug','STlik':False}['STlik'] # True; 'aug'; False
    var_out = 'cov' # True; 'cov'; False
    STlik = True
    lrz = Lorenz(num_traj=num_traj, prior_params=prior_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik)
    temp_dim,spat_dim=lrz.misfit.obs[0].shape
    
    
    
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    
    # define the emulator (DNN)
    
    # load data
    ensbl_sz = 500
    folder = './train_NN'
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY_stlik.npz'))
    X=loaded['X']
    Y=loaded['Y']
    Y=Y.reshape((-1,temp_dim,spat_dim))
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    scaler = MinMaxScaler()
    # transform data
    X = scaler.fit_transform(X)
    # pre-processing: scale X to 0-1
    #X-=np.nanmin(X,axis=np.arange(X.ndim)[1:])
    #X/=np.nanmax(X,axis=np.arange(X.ndim)[1:])
    
    # split train/test
    num_samp=X.shape[0]
    #n_tr=np.int(num_samp*.75)
    #x_train,y_train=X[:n_tr],Y[:n_tr]
    #x_test,y_test=X[n_tr:],Y[n_tr:]
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    y_train,y_test=Y[tr_idx],Y[te_idx]
    
    # define DNN-RNN
    depth=args.depth
    #activations={'hidden':'relu','latent':'linear','output':'linear','lstm':'tanh'}
    activations={'hidden':tf.keras.layers.LeakyReLU(alpha=.01),'latent':'linear','output':'sigmoid','lstm':'linear'}#
    #activations={'conv':tf.math.sin,'latent':tf.math.sin,'output':'linear','lstm':'tanh'}
    latent_dim=y_train.shape[2]
    droprate=args.droprate
    #kernel_initializers={'conv':sin_init,'latent':sin_init,'output':'glorot_uniform'}
    kernel_initializers={'hidden':'he_uniform','latent':'glorot_uniform','output':'glorot_uniform'}
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lrrate,amsgrad=True,clipnorm=1)###clipnorm=1.?????????????????####
    
    dnnrnn=DNN_RNN(x_train.shape[1], y_train.shape[1:], depth=depth, latent_dim=latent_dim, droprate=droprate,
                   activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer)
    # =============================================================================
    # #dnnrnn.model.trainable_variables
    # tf.compat.v1.disable_eager_execution()
    # tf.enable_eager_execution()
    # sess=tf.compat.v1.InteractiveSession()
    # 
    # from tf.keras import backend as k
    # listOfVariableTensors = dnnrnn.model.trainable_weights
    # outputTensor = dnnrnn.model.output
    # gradients = k.gradients(outputTensor, listOfVariableTensors)
    # 
    # #sess.run(tf.compat.v1.global_variables_initializer())
    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     evaluated_gradients = sess.run(gradients,feed_dict={dnnrnn.model.input:x_train})
    # print(evaluated_gradients)
    # =============================================================================
    
    
    # custom_loss = lambda y_true, y_pred: [tf.square(loglik(y_true)-loglik(y_pred)), (y_true-y_pred)/lrz.misfit.noise_variance]
    savepath=folder+'/DNN_RNN/saved_model'
    if not os.path.exists(savepath): os.makedirs(savepath)
    import time
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    f_name='dnn_rnn_'+algs[alg_no]+str(ensbl_sz)+'_'+ctime
    
    try:
    #     dnnrnn.model=load_model(os.path.join(savepath,f_name+'.h5'),custom_objects={'loss':None})
        dnnrnn.model.load_weights(os.path.join(savepath,f_name+'.h5'))
        print(f_name+' has been loaded!')
    except Exception as err:
        print(err)
        print('Train DNN-RNN...\n')
        epochs=100
        patience=args.patience
        import timeit
        t_start=timeit.default_timer()
        dnnrnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training DNN-RNN: {}'.format(t_used))
        # save DNN-RNN
    #         dnnrnn.model.save('./result/dnnrnn_model.h5')
    #         dnnrnn.save('./result','dnnrnn_'+algs[alg_no])
    #    dnnrnn.model.save_weights(os.path.join(savepath,f_name+'.h5'))#'./result','dnnrnn_'+algs[alg_no]+'.h5'
    
    #W = tf.convert_to_tensor(lrz.misfit.stgp.tomat(),dtype=tf.float32) # lrz.misfit.stgp.tomat()
    #loglik = lambda y: -0.5*tf.math.reduce_sum(tf.reshape(y-lrz.misfit.obs,[-1,output_dim])*tf.transpose(tf.linalg.solve(W,tf.transpose(tf.reshape(y-lrz.misfit.obs,[-1,output_dim])))),axis=1)
    #logLik = lambda x: loglik(dnnrnn.model(x))
    logLik = lambda x: -lrz.misfit.cost(obs=dnnrnn.model(x))#lambda x: -0.5*tf.math.reduce_sum((dnnrnn.model(x)-lrz.misfit.obs)**2/lrz.misfit.nzvar,axis=[1,2])
    
    # select some gradients to evaluate and compare
    import timeit
    t_used = np.zeros(2)
    
    n_dif = 100
    dif = np.zeros((n_dif,2))
    prng=np.random.RandomState(2020)
    sel4eval = prng.choice(num_samp,size=n_dif,replace=False)
    X=X[sel4eval]; Y=Y[sel4eval]
    sel4print = prng.choice(n_dif,size=10,replace=False)
    prog=np.ceil(n_dif*(.1+np.arange(0,1,.1)))
    
    
    for n in range(n_dif):
        u=X[n]
        # calculate gradient
        t_start=timeit.default_timer()
        ll_xact,dll_xact = lrz.get_geom(u,[0,1])[:2]
        t_used[0] += timeit.default_timer()-t_start
        # emulate gradient
        t_start=timeit.default_timer()
        ll_emul = logLik(u[None,:])
        #dll_emul = dnnrnn.gradient(u[None,:], logLik)
        t_used[1] += timeit.default_timer()-t_start
        # test difference
        dif_fun = np.abs(ll_xact - ll_emul)
        #dif_grad = dll_xact - dll_emul
        dif[n] = np.array([dif_fun, np.linalg.norm(dif_fun)/np.linalg.norm(ll_xact)])#, np.linalg.norm(dif_fun)/np.linalg.norm(ll_xact)
        
    #     # check the gradient extracted from emulation
    #     v=lrz.prior.sample()
    #     h=1e-4
    #     dll_emul_fd_v=(logLik(u[None,:]+h*v[None,:])-logLik(u[None,:]))/h
    #     reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v))/np.linalg.norm(v)
    #     print('Relative difference between finite difference and extracted results: {}'.format(reldif))
        
        if n+1 in prog:
            print('{0:.0f}% evaluation has been completed.'.format(np.float(n+1)/n_dif*100))
        
        
    print('Time used to calculate vs emulate loglik: {} vs {}'.format(*t_used.tolist()))
    # save to file
    import pandas as pd
    savepath=folder+'/DNN_RNN/summary'
    if not os.path.exists(savepath): os.makedirs(savepath)
    #file=os.path.join(savepath,'dif-'+ctime+'.txt')
    #np.savetxt(file,dif)
    depth=str(depth)
    act_str=','.join([val.__name__ if type(val).__name__=='function' else val.name if callable(val) else val for val in activations.values()])
    dif_fun_sumry=[dif[:,0].min(),np.median(dif[:,0]),dif[:,0].max()]
    dif_fun_str=np.array2string(np.array(dif_fun_sumry),precision=2,separator=',').replace('[','').replace(']','') # formatter={'float': '{: 0.2e}'.format}
    remdif_fun_sumry=[dif[:,1].min(),np.median(dif[:,1]),dif[:,1].max()]
    remdif_fun_str=np.array2string(np.array(remdif_fun_sumry),precision=2,separator=',').replace('[','').replace(']','') 
    sumry_header=('Time','activations','depth','droprate','learning_rate','patience','dif_fun (min,med,max)','remdif_fun (min,med,max)')
    sumry_np=np.array([ctime,act_str,depth,droprate,args.lrrate,patience,dif_fun_str,remdif_fun_str])
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
        
    import seaborn as sns
    import matplotlib.pyplot as plt
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
    para0 = lrz.misfit.true_params
    marg = [1,.2,1]; res = 100
    grid_data = lrz.misfit.true_params.copy()
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

if __name__ == '__main__':
    main()
