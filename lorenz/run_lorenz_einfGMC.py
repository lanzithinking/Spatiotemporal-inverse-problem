"""
Main function to run (inf-)geometric MCMC for Lorenz63 inverse problem
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# the inverse problem
from Lorenz import Lorenz

# MCMC
import sys
sys.path.append( "../" )
from nn.dnn import DNN
from nn.dnn_rnn import DNN_RNN
from sampler.einfGMC import einfGMC

# relevant geometry
from geom_emul import geom

np.set_printoptions(precision=3, suppress=True)
seed=2021
np.random.seed(seed)
tf.random.set_seed(seed)

def main(seed=2021):
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('emuNO', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=10000)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[3.5,3.5,3.,None,None]) # .00001 for simple likelihood model
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=['e'+a for a in ('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC')])
    parser.add_argument('emus', nargs='?', type=str, default=['dnn','dnnrnn'])
    args = parser.parse_args()
    
    # likelihood model
    mdls=('simple','STlik')
    num_mdls=len(mdls)
    mdl_no=args.emuNO # DNN for simple model; DNN-LSTM for spatiotemporal model
    ## define Lorenz63 inverse problem ##
    num_traj = 1
    t_init = 100
    t_final = 110
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = {'simple':'aug','STlik':False}[mdls[mdl_no]] # True; 'aug'; False
    var_out = 'cov' # True; 'cov'; False
    STlik = (mdls[mdl_no]=='STlik')
    lrz = Lorenz(num_traj=num_traj, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik) # set STlik=False for simple likelihood; STlik has to be used with avg_traj
    
    ##------ define networks ------##
    # training data algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    # load data
    ensbl_sz = 500
    folder = './train_NN'
    
    ##---- EMULATOR ----##
    # prepare for training data
    loaded=np.load(file=os.path.join(folder,mdls[mdl_no]+'_'+algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
    X=loaded['X']
    Y=loaded['Y']
    num_samp=X.shape[0]
#     n_tr=np.int(num_samp*.75)
#     x_train,y_train=X[:n_tr],Y[:n_tr]
#     x_test,y_test=X[n_tr:],Y[n_tr:]
    tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
    te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
    x_train,x_test=X[tr_idx],X[te_idx]
    y_train,y_test=Y[tr_idx],Y[te_idx]
    # define emulator
    input_dim=len(lrz.ode_params)
    if args.emus[args.emuNO]=='dnn':
        output_dim=lrz.misfit.obs.size
        depth=4
        node_sizes=[input_dim,30,100,output_dim]
        activations={'hidden':tf.keras.layers.LeakyReLU(alpha=0.01),'output':'linear'}
        droprate=0.0
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005,amsgrad=True)
        W = tf.convert_to_tensor(lrz.misfit.nzvar[0],dtype=tf.float32)
        custom_loss = lambda y_true, y_pred: [tf.math.reduce_sum((y_true-y_pred)*tf.transpose(tf.linalg.solve(W,tf.transpose(y_true-y_pred))),axis=1), None]
        emulator=DNN(x_train.shape[1], y_train.shape[1], depth=depth, node_sizes=node_sizes, droprate=droprate,
                     activations=activations, optimizer=optimizer, loss=custom_loss)
    elif args.emus[args.emuNO]=='dnnrnn':
        temp_dim,spat_dim=lrz.misfit.obs.shape[1:]
        y_train=y_train.reshape((-1,temp_dim,spat_dim)); y_test=y_test.reshape((-1,temp_dim,spat_dim))
        output_dim = np.prod(lrz.misfit.obs.shape[1:])
        depth=4
        node_sizes=[input_dim,30,100,output_dim]
        activations={'hidden':'softplus','output':'sigmoid','gru':'linear'}
        droprate=.5
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005,amsgrad=True)
        W = tf.convert_to_tensor(lrz.misfit.stgp.tomat(),dtype=tf.float32)
        custom_loss = lambda y_true, y_pred: [tf.math.reduce_sum(tf.reshape(y_true-y_pred,[-1,output_dim])*tf.transpose(tf.linalg.solve(W,tf.transpose(tf.reshape(y_true-y_pred,[-1,output_dim])))),axis=1), None]
        emulator=DNN_RNN(x_train.shape[1], y_train.shape[1:], depth=depth, node_sizes=node_sizes, droprate=droprate,
                        activations=activations, optimizer=optimizer, loss=custom_loss)
    f_name=args.emus[args.emuNO]+'_'+algs[alg_no]+'_'+str(ensbl_sz)
    # load emulator
    try:
        emulator.model=load_model(os.path.join(folder,f_name+'.h5'),custom_objects={'loss':None})
        print(f_name+' has been loaded!')
    except:
        try:
            emulator.model.load_weights(os.path.join(folder,f_name+'.h5'))
            print(f_name+' has been loaded!')
        except:
            print('\nNo emulator found. Training {}...\n'.format(args.emus[args.emuNO]))
            epochs=500
            patience=5
            emulator.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1,patience=patience)
            # save emulator
            try:
                emulator.model.save(os.path.join(folder,f_name+'.h5'))
            except:
                emulator.model.save_weights(os.path.join(folder,f_name+'.h5'))
    
    # initialization
    u0=lrz.prior.sample()
    emul_geom=lambda q,geom_ord=[0],**kwargs:geom(q,lrz,emulator,geom_ord,**kwargs)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    e_infGMC=einfGMC(u0,lrz,emul_geom,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO])#,k=5) # uncomment for manifold algorithms
    mc_fun=e_infGMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append ODE information including the count of solving
    filename_=os.path.join(e_infGMC.savepath,e_infGMC.filename+'.pckl')
    filename=os.path.join(e_infGMC.savepath,'Lorenz63_'+{True:'avg',False:'full','aug':'avgaug'}[lrz.misfit.avg_traj]+'_'+e_infGMC.filename+'_'+args.emus[args.emuNO]+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    soln_count=lrz.ode.soln_count
    pickle.dump([num_traj,obs_times,avg_traj,soln_count,args],f)
    f.close()

if __name__ == '__main__':
    main()
    # collect samples from multiple runs with different random seeds
    # n_seed = 10; i=0; n_success=0
    # while n_success < n_seed:
    #     i+=1
    #     seed_i=2021+i*10
    #     try:
    #         print("Running for seed %d ...\n"% (seed_i))
    #         main(seed=seed_i)
    #         n_success+=1
    #     except Exception as e:
    #         print(e)
    #         pass