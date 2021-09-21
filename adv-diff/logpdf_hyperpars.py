#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log-pdf of hyper-parameters of STGP models
------------------------------------------
@author: Shuyi Li
"""
import numpy as np

def logpost_eta(eta, inf_GMC, m, V, opt=[0]):
    '''
    compute the log-posterior of eta
    option 0: x; option 1: t
    '''
    if 0 in opt:
        inf_GMC.model.misfit.stgp.update(C_x=inf_GMC.model.misfit.stgp.C_x.update(l = np.exp(eta)))
    elif 1 in opt:
        inf_GMC.model.misfit.stgp.update(C_t=inf_GMC.model.misfit.stgp.C_t.update(l = np.exp(eta)))
    # loglik = inf_GMC.geom(inf_GMC.q)[0]
    loglik = -inf_GMC.model.misfit.cost(inf_GMC.model.x)
    logpri = -.5*(eta-m)**2/V
    logpost = loglik + logpri
    return logpost
