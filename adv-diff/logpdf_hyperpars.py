#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log-pdf of hyper-parameters of STGP models
------------------------------------------
@author: Shuyi Li
"""
import numpy as np

def logpost_eta(eta, inf_GMC, m, V, opt=[0], **kwargs):
    '''
    compute the log-posterior of eta
    option 0: x; option 1: t
    '''
    a = kwargs.get('a');  b = kwargs.get('b')
    int_sigma2 = all([k is not None for k in [a,b]]) # whether to integrate out sigma2
    if type(eta) is not np.ndarray:
        eta = np.asarray([eta,])
    if 0 in opt:
        inf_GMC.model.misfit.stgp.update(C_x=inf_GMC.model.misfit.stgp.C_x.update(l = np.exp(eta[0])))
    if 1 in opt:
        inf_GMC.model.misfit.stgp.update(C_t=inf_GMC.model.misfit.stgp.C_t.update(l = np.exp(eta[-1])))
    if int_sigma2:
        sigma2_t = inf_GMC.model.misfit.stgp.C_t.sigma2
        inf_GMC.model.misfit.stgp.update(C_t=inf_GMC.model.misfit.stgp.C_t.update(sigma2=1))
    if not int_sigma2:
        # loglik = inf_GMC.geom(inf_GMC.q)[0]
        loglik = -inf_GMC.model.misfit.cost(inf_GMC.model.x)
    else:
        nll, quad = inf_GMC.model.misfit.cost(inf_GMC.model.x, 'both')
        loglik = - (nll-quad) - (a+inf_GMC.model.misfit.stgp.N/2)*np.log(b+quad)
        inf_GMC.model.misfit.stgp.update(C_t=inf_GMC.model.misfit.stgp.C_t.update(sigma2=sigma2_t))
    logpri = np.sum(-.5*(eta-m)**2/V)
    logpost = loglik + logpri
    return logpost
