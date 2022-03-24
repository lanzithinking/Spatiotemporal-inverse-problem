#!/usr/bin/env python
"""
Dense Neural Network (input) - Recurrent (output) Neural Network (input)
Shiwei Lan @ASU, 2022
--------------------------
DNN-RNN in TensorFlow 2.8
-------------------------
Created March 13, 2022
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dropout,Dense,Reshape,SimpleRNN,GRU,LSTM
from tensorflow.keras.models import Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping


class DNN_RNN:
    def __init__(self, input_dim, output_shape, depth=3, **kwargs):
        """
        Dense-Recurrent Neural Network
        --------------------------------------------------------------------------------
        input_dim: the dimension of the input space
        output_shape: output shape (temp_dim, spat_dim)
        depth: the depth of the network
        node_sizes: sizes of the nodes of the network, which can override depth
        droprate: the rate of Dropout
        activations: specification of activation functions, can be a list of strings or Keras activation layers
        kernel_initializers: kernel_initializers corresponding to activations
        """
        self.input_dim=input_dim
        self.output_shape=output_shape
        self.output_dim=np.prod(self.output_shape)
        self.depth = depth
        self.node_sizes = kwargs.pop('node_sizes',None)
        if self.node_sizes is None:
            self.node_sizes = np.linspace(self.input_dim,self.output_dim,self.depth+1,dtype=np.int)
        else:
            self.depth=np.size(self.node_sizes)-1
        if self.node_sizes[0]!=self.input_dim or self.node_sizes[-1]!=self.output_dim:
            raise ValueError('Node sizes not matching input/output dimensions!')
        self.droprate = kwargs.pop('droprate',0)
        self.activations = kwargs.pop('activations',{'hidden':'relu','output':'sigmoid','lstm':'tanh'})
        self.kernel_initializers=kwargs.pop('kernel_initializers',{'hidden':'glorot_uniform','output':'glorot_uniform'})
        # build neural network
        self.build(**kwargs)
    
    def _set_layers(self, input):
        """
        Set network layers
        """
        output=input
        for i in range(self.depth):
            ker_ini = self.kernel_initializers['hidden'](output.shape[1]*30**(i==0)) if callable(self.kernel_initializers['hidden']) else self.kernel_initializers['hidden']
            if callable(self.activations['hidden']):
                output=Dense(units=self.node_sizes[i+1], kernel_initializer=ker_ini, name='hidden_{}'.format(i))(output)
                output=self.activations['hidden'](output)
            else:
                output=Dense(units=self.node_sizes[i+1], activation=self.activations['hidden'], kernel_initializer=ker_ini, name='hidden_{}'.format(i))(output)
            if self.droprate>0: output=Dropout(rate=self.droprate)(output)
        ker_ini = self.kernel_initializers['output'](output.shape[1]) if callable(self.kernel_initializers['output']) else self.kernel_initializers['output']
        # transform original output(n_batch,temp_dim*spat_dim) to RNN input(n_batch,temp_dim,spat_dim)
        output = Reshape(self.output_shape)(output)
        recurr = {'output':SimpleRNN,'gru':GRU,'lstm':LSTM}[list(self.activations.keys())[-1]]
        if len(self.activations)==2:
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
            # L+=tf.math.reduce_sum(tf.math.reduce_sum(self.batch_jacobian()*loss_f(y_true,y_pred)[1][:,:,None],axis=1)**2,axis=[1]) # diff in gradient potential
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
                                      shuffle=False,
                                      callbacks=[es],
                                      verbose=verbose, **kwargs)
    
    def save(self, savepath='./',filename='dnnrnn_model'):
        """
        Save the trained model for future use
        """
        import os
        self.model.save(os.path.join(savepath,filename+'.h5'))
    
    def evaluate(self, input):
        """
        Output model prediction
        """
        assert input.shape[1]==self.input_dim, 'Wrong input dimension!'
        return self.model.predict(input)
    
    def gradient(self, input, objf=None):
        """
        Obtain gradient of objective function wrt input
        """
        if not objf:
            #where do we define self.y_train
            objf = lambda x: tf.keras.losses.MeanSquaredError(self.y_train,self.model(x))
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
    sys.path.append("../Lorenz/")
    from Lorenz import Lorenz
    # set random seed
    seed=2021
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # define the inverse problem
    num_traj = 1 # only consider single trajectory!
    prior_params = {'mean':[2.0, 1.2, 3.3], 'std':[0.2, 0.5, 0.15]}
    t_init = 100
    t_final = 110
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = False # True; 'aug'; False
    var_out = 'cov' # True; 'cov'; False
    STlik = 'sep'
    lrz = Lorenz(num_traj=num_traj, prior_params=prior_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik)
    temp_dim,spat_dim=lrz.misfit.obs.shape[1:]
    # algorithms
    algs=['EKI','EKS']
    num_algs=len(algs)
    alg_no=1
    mdls=('simple','STlik')
    num_mdls=len(mdls)
    mdl_no=1 # DNN-RNN for spatiotemporal model
    
    # load data
    ensbl_sz = 500
    folder = '../Lorenz/train_NN'
    loaded=np.load(file=os.path.join(folder,mdls[mdl_no]+'_'+algs[alg_no]+'_ensbl'+str(ensbl_sz)+'_training_XY.npz'))
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
    depth=4
    node_sizes=[3,30,100,3]
    activations={'hidden':'relu','output':'linear','lstm':'tanh'}
    # activations={'hidden':'relu','output':tf.keras.layers.PReLU(),'lstm':'tanh'}
#     activations={'hidden':tf.math.sin,'output':'linear','lstm':'tanh'}
    droprate=.25
    sin_init=lambda n:tf.random_uniform_initializer(minval=-tf.math.sqrt(6/n), maxval=tf.math.sqrt(6/n))
    #kernel_initializers={'hidden':sin_init,'output':'glorot_uniform'}
    kernel_initializers={'hidden':'he_uniform','output':'glorot_uniform'}
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    dnnrnn=DNN_RNN(x_train.shape[1], y_train.shape[1:], depth=depth, node_sizes=node_sizes, droprate=droprate,
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
        # save DNN-RNN
#         dnnrnn.model.save('./result/dnnrnn_model.h5')
#         dnnrnn.save('./result','dnnrnn_'+algs[alg_no])
        dnnrnn.model.save_weights('./result','dnnrnn_'+algs[alg_no]+'.h5')
    
    # some more test
    loglik = lambda x: -0.5*lrz.misfit.cost(obs=dnnrnn.model(x))
    import timeit
    t_used = np.zeros((1,2))
    for n in range(10):
        u=lrz.prior.sample()
        # calculate gradient
        t_start=timeit.default_timer()
        ll_xact,dll_xact = lrz.get_geom(u,[0,1])[:2]
        t_used[0] += timeit.default_timer()-t_start
        # emulate gradient
        t_start=timeit.default_timer()
        ll_emul = logLik(u[None,:]).numpy()
        dll_emul = dnnrnn.gradient(u[None,:], loglik)
        t_used[1] += timeit.default_timer()-t_start
        # test difference
        dif_fun = np.abs(ll_xact - ll_emul)
        dif_grad = dll_xact - dll_emul
        dif[n] = np.array([dif_fun, np.linalg.norm(dif_grad)/np.linalg.norm(dll_xact)])
        print('Difference between the calculated and emulated gradients: min ({}), med ({}), max ({})'.format(dif.min(),np.median(dif.get_local()),dif.max()))
        
    print('Time used to calculate vs emulate gradients: {} vs {}'.format(t_used))
    