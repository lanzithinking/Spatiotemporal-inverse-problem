#!/usr/bin/env python
"""
Deep (input) - Recurrent (output) Neural Network (input)
--------------------------
DNN-RNN in TensorFlow 2.2
-------------------------
Created December 21, 2020
"""
__author__ = "Shuyi Li"
__copyright__ = "Copyright 2022"
__credits__ = "Shiwei Lan"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Reshape,SimpleRNN,GRU,LSTM
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class DNN_RNN:
    def __init__(self, input_dim, output_shape, depth=3, **kwargs):
        """
        Deep Neural Network
        --------------------------------------------------------------------------------
        input_dim: the dimension of the input space
        output_shape: the dimension of the output space (temp_dim, spat_dim)
        depth: the depth of the network
        node_sizes: sizes of the nodes of the network, which can override depth
        droprate: the rate of Dropout
        activations: specification of activation functions, can be a list of strings or Keras activation layers
        kernel_initializers: kernel_initializers corresponding to activations
        """
        self.input_dim=input_dim
        self.output_shape=output_shape
        self.latent_dim = kwargs.pop('latent_dim',self.output_shape[1])
        self.depth = depth
        self.node_sizes = kwargs.pop('node_sizes',None)
        if self.node_sizes is None:
            self.node_sizes = np.linspace(self.input_dim,self.output_shape[0]*self.latent_dim,self.depth+1,dtype=np.int)
        else:
            self.depth=np.size(self.node_sizes)-1
        if self.node_sizes[0]!=self.input_dim or self.node_sizes[-1]!=self.output_shape[0]*self.latent_dim:
            raise ValueError('Node sizes not matching input/output dimensions!')
        self.droprate = kwargs.pop('droprate',0)
        self.activations = kwargs.pop('activations',{'hidden':'relu','latent':'linear', 'output':'sigmoid','lstm':'tanh'})
        self.kernel_initializers=kwargs.pop('kernel_initializers',{'hidden':'glorot_uniform','latent':'glorot_uniform','output':'glorot_uniform'})
        # build neural network
        self.build(**kwargs)
        
        
    
    def _set_layers(self, input):
        """
        Set network layers
        """
        output=input
        for i in range(self.depth):
            layer_name = 'latent' if i==self.depth-1 else 'hidden_layer{}'.format(i)
            activation = self.activations['latent'] if i==self.depth-1 else self.activations['hidden']
            ker_ini_hidden = self.kernel_initializers['hidden'](output.shape[1]*30**(i==0)) if callable(self.kernel_initializers['hidden']) else self.kernel_initializers['hidden']
            ker_ini_latent = self.kernel_initializers['latent'](output.shape[1]) if callable(self.kernel_initializers['latent']) else self.kernel_initializers['latent']
            ker_ini = ker_ini_latent if i==self.depth-1 else ker_ini_hidden
            if callable(activation):
                output=Dense(units=self.node_sizes[i+1], kernel_initializer=ker_ini, name=layer_name)(output)
                output=activation(output)
            else:
                output=Dense(units=self.node_sizes[i+1], activation=activation, kernel_initializer=ker_ini, name=layer_name)(output)
            if self.droprate>0: output=Dropout(rate=self.droprate)(output)
        
        ker_ini = self.kernel_initializers['output'](output.shape[1]) if callable(self.kernel_initializers['output']) else self.kernel_initializers['output']
        # transform original output(n_batch,temp_dim*spat_dim) to RNN input(n_batch,temp_dim,spat_dim)
        output = Reshape((self.output_shape[0],self.latent_dim))(output)
        recurr = {'output':SimpleRNN,'gru':GRU,'lstm':LSTM}[list(self.activations.keys())[-1]]
        if len(self.activations)==3:
            output = recurr(units=self.output_shape[1], return_sequences=True,  activation=self.activations['output'],
                            kernel_initializer=ker_ini, name='recur')(output)
        else:
            output = recurr(units=self.output_shape[1], return_sequences=True,  activation=list(self.activations.values())[-1], recurrent_activation=self.activations['output'],
                            kernel_initializer=ker_ini, name='recur')(output)
        return output

    
    def _custom_loss(self,loss_f):
        """
        Wrapper to customize loss function (on latent space)
        """
        def loss(y_true, y_pred):
#             L=tf.keras.losses.MSE(y_true, y_pred)
            L=loss_f(y_true,y_pred)[0] # diff in potential
            L+=tf.math.reduce_sum(tf.math.reduce_sum(self.batch_jacobian()*loss_f(y_true,y_pred)[1][:,:,None,None,None],axis=1)**2,axis=[1,2,3]) # diff in gradient potential
            return L
        return loss

    def build(self,**kwargs):
        """
        Set up the network structure and compile the model with optimizer, loss and metrics.
        """
        # initialize model
        input = Input(shape=self.input_dim, name='input')
        # set model layers
        output = self._set_layers(input)
        self.model = Model(input, output, name='dnn_rnn')
        # compile model
        optimizer = kwargs.pop('optimizer','adam')
        loss = kwargs.pop('loss','mse')
        metrics = kwargs.pop('metrics',['mae'])
        self.model.compile(optimizer=optimizer, loss=self._custom_loss(loss) if callable(loss) else loss, metrics=metrics, **kwargs)
    
    def train(self, x_train, y_train, x_test=None, y_test=None, batch_size=32, epochs=100, verbose=0, **kwargs):
        """
        Train the model with data
        """
        num_samp=x_train.shape[0]
        if any([i is None for i in (x_test, y_test)]):
            tr_idx=np.random.choice(num_samp,size=np.floor(.75*num_samp).astype('int'),replace=False)
            te_idx=np.setdiff1d(np.arange(num_samp),tr_idx)
            x_test, y_test = x_train[te_idx], y_train[te_idx]
            x_train, y_train = x_train[tr_idx], y_train[tr_idx]
        patience = kwargs.pop('patience',0)
        es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=patience)
        self.history = self.model.fit(x_train, y_train,
                                      validation_data=(x_test, y_test),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      callbacks=[es],
                                      verbose=verbose, **kwargs)
        
    
    def save(self, savepath='./',filename='dnn_rnn_model'):
        """
        Save the trained model for future use
        """
        import os
        self.model.save(os.path.join(savepath,filename+'.h5'))
    
    def evaluate(self, input):
        """
        Output model prediction
        """
        assert input.shape[1]==self.input_dim, 'Wrong image shape!'
        return self.model.predict(input)
    
    def gradient(self, input, objf=None, y_train=None):
        """
        Obtain gradient of objective function wrt input
        """
        if not objf:
            #where do we define self.y_train
            objf = tf.keras.losses.MeanSquaredError()
            objf = lambda x: objf(y_train,self.model(x))
        x = tf.Variable(input, trainable=True, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            obj = objf(x)
        grad = tape.gradient(obj,x).numpy()
        return np.squeeze(grad)
    
    def jacobian(self, input):
        """
        Obtain Jacobian matrix of output wrt input
        """
#         x = tf.constant(input)
        x = tf.Variable(input, trainable=True, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = self.model(x)
        jac = g.jacobian(y,x).numpy()
        return np.squeeze(jac)
    
    def batch_jacobian(self, input=None):
        """
        Obtain Jacobian matrix of output wrt input
        ------------------------------------------
        Note: when using model input, it has to run with eager execution disabled in TF v2.2.0
        """
        if input is None:
            x = self.model.input
        else:
#             x = tf.constant(input)
            x = tf.Variable(input, trainable=True, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = self.model(x)
        jac = g.batch_jacobian(y,x)
        return jac if input is None else np.squeeze(jac.numpy())

if __name__ == '__main__':
    import sys,os
    sys.path.append( "../" )
    from lorenz.lorenz import Lorenz
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
    var_out = True # True; 'cov'; False
    STlik = True
    lrz = Lorenz(num_traj=num_traj, prior_params=prior_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik) # set STlik=False for simple likelihood; STlik has to be used with avg_traj
        
    temp_dim,spat_dim=lrz.misfit.obs[0].shape
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    
    # load data
    ensbl_sz = 500
    folder = './train_NN'
    loaded=np.load(file=os.path.join(folder,algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
    X=loaded['X']
    Y=loaded['Y']
    Y=Y.reshape((-1,temp_dim,spat_dim))
    # pre-processing: scale X to 0-1
    #X-=np.nanmin(X,axis=np.arange(X.ndim)[1:])
    #X/=np.nanmax(X,axis=np.arange(X.ndim)[1:])
    
    # split train/test
    num_samp=X.shape[0]
    n_tr=np.int(num_samp*.75)
    x_train,y_train=X[:n_tr],Y[:n_tr]
    x_test,y_test=X[n_tr:],Y[n_tr:]
    
    # define DNN-RNN
    depth=2
    #activations={'conv':'relu','latent':tf.keras.layers.PReLU(),'output':'linear','lstm':'tanh'}
    activations={'hidden':'relu','latent':'linear','output':'linear','lstm':'tanh'}
#     activations={'conv':tf.math.sin,'latent':tf.math.sin,'output':'linear','lstm':'tanh'}
    latent_dim=y_train.shape[2]
    droprate=.25
    sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
    #kernel_initializers={'conv':sin_init,'latent':sin_init,'output':'glorot_uniform'}
    kernel_initializers={'hidden':'he_uniform','latent':sin_init,'output':'glorot_uniform'}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    dnnrnn=DNN_RNN(x_train.shape[1], y_train.shape[1:], depth=depth, latent_dim=latent_dim, droprate=droprate,
                   activations=activations, kernel_initializers=kernel_initializers, optimizer=optimizer)
    try:
#         dnnrnn.model=load_model('./result/dnnrnn_'+algs[alg_no]+'.h5')
        dnnrnn.model.load_weights('./result/dnnrnn_'+algs[alg_no]+'.h5')
        print('dnnrnn_'+algs[alg_no]+'.h5'+' has been loaded!')
    except Exception as err:
        print(err)
        print('Train DNN-RNN...\n')
        epochs=100
        import timeit
        t_start=timeit.default_timer()
        dnnrnn.train(x_train,y_train,x_test=x_test,y_test=y_test,epochs=epochs,batch_size=64,verbose=1)
        t_used=timeit.default_timer()-t_start
        print('\nTime used for training DNN-RNN: {}'.format(t_used))
        # save CNN-RNN
#         dnnrnn.model.save('./result/dnnrnn_model.h5')
#         dnnrnn.save('./result','dnnrnn_'+algs[alg_no])
        dnnrnn.model.save_weights('./result','dnnrnn_'+algs[alg_no]+'.h5')
    
    # some more test
    loglik = lambda x: -0.5*tf.math.reduce_sum((dnnrnn.model(x)-lrz.misfit.obs[0])**2/lrz.misfit.nzvar[0],axis=[1,2])
    import timeit
    t_used = np.zeros((1,2))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,6), facecolor='white')
    plt.ion()
    plt.show(block=True)
    
    for n in range(10):
        u=lrz.prior.sample()
        # calculate gradient
        t_start=timeit.default_timer()
        ll_xact,dll_xact = lrz.get_geom(u,[0,1])[:2]
        t_used[0] += timeit.default_timer()-t_start
        # emulate gradient
        t_start=timeit.default_timer()
        
        dll_emul = dnnrnn.gradient(u[None,:], loglik)
        t_used[1] += timeit.default_timer()-t_start
        # test difference
        ll_emul = loglik(u)
        dif_fun = np.abs(ll_xact - ll_emul)
        
        dif_grad = dll_xact - dll_emul
        dif = np.array([dif_fun, np.linalg.norm(dif_grad)/np.linalg.norm(dll_xact)])
        print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})'.format(dif[1].min(),np.median(dif[1]),dif[1].max()))
        print('Difference between the calculated and emulated loglik: ({})'.format(dif))
#         # check the gradient extracted from emulation
#         v=lrz.prior.sample()
#         h=1e-4
#         dll_emul_fd_v=(logLik(u[None,:]+h*v[None,:])-logLik(u[None,:]))/h
#         reldif = abs(dll_emul_fd_v - dll_emul.flatten().dot(v))/np.linalg.norm(v)
#         print('Relative difference between finite difference and extracted results: {}'.format(reldif))
        
        # plot
        plt.subplot(121)
        plt.plot(dll_xact)
        plt.title('Calculated Gradient')
        plt.subplot(122)
        plt.plot(dll_emul)
        plt.title('Emulated Gradient')
        plt.draw()
        plt.pause(1.0/30.0)
        
    print('Time used to calculate vs emulate gradients: {} vs {}'.format(t_used[0], t_used[1]))
