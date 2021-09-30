"""
Modified geometry function for STGP with sigma2 integrated out
--------------------------------------------------------------
@author: Shiwei Lan @ ASU
Created: Sept. 2021
"""
import numpy as np
STATE=0; PARAMETER=1

def geom(parameter,a,b,model,geom_ord=[0],whitened=False):
    model.misfit.stgp.update(C_t=model.misfit.stgp.C_t.update(sigma2=1))
    loglik=None; agrad=None;
    
    # un-whiten if necessary
    if whitened:
        parameter=model.whtprior.v2u(parameter)
    
    # get log-likelihood
    if any(s>=0 for s in geom_ord):
        model.x[PARAMETER] = parameter
        model.pde.solveFwd(model.x[STATE], model.x)
        nll, quad = model.misfit.cost(model.x, option='both')
        a1=a+model.misfit.stgp.N/2; b1=b+quad
        loglik = - (nll-quad) - a1*np.log(b1)
    
    # get gradient
    if any(s>=1 for s in geom_ord):
        agrad = -model._get_grad(parameter)*(a1/b1)
        if whitened:
            agrad_ = agrad.copy(); agrad.zero()
            model.whtprior.C_act(agrad_,agrad,comp=0.5,transp=True)
    
    return loglik,agrad,np.nan,np.nan