"""
Geometric functions by emulator emulation
"""

import numpy as np
import tensorflow as tf

def geom(parameter,bip,emulator,geom_ord=[0],**kwargs):
    if parameter is None:
        parameter=bip.prior.mean
    loglik=None; gradlik=None; HessApply=None; eigs=None
    
    param = parameter[None,:]
    
    W = tf.convert_to_tensor(bip.misfit.stgp.tomat() if bip.misfit.STlik else bip.misfit.nzvar[0],dtype=tf.float32)
    # ll_f = lambda x: -0.5*tf.math.reduce_sum(tf.reshape(emulator.model(x)-bip.misfit.obs,[-1,bip.misfit.obs.size])*tf.transpose(tf.linalg.solve(W,tf.transpose(tf.reshape(emulator.model(x)-bip.misfit.obs,[-1,bip.misfit.obs.size])))),axis=1)
    def ll_f(x):
        dif_obs = emulator.model(x)-bip.misfit.obs
        if bip.misfit.STlik: dif_obs = tf.reshape(dif_obs,[-1,bip.misfit.obs.size])
        ll = -0.5*tf.math.reduce_sum(dif_obs*tf.transpose(tf.linalg.solve(W,tf.transpose(dif_obs))),axis=1)
        return ll
    
    # get log-likelihood
    if any(s>=0 for s in geom_ord):
        loglik = ll_f(param).numpy()
    
    # get gradient
    if any(s>=1 for s in geom_ord):
        gradlik = emulator.gradient(param, ll_f)
    
    if any(s>=1.5 for s in geom_ord):
        pass
    
    if any(s>1 for s in geom_ord) and len(kwargs)!=0:
        pass
    
    return loglik,gradlik,HessApply,eigs