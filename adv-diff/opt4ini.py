#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization to provide better initialization for MCMC
------------------------------------------------------
Created on Sat Sep  4 15:41:55 2021
@author: Shuyi Li
"""
import time, copy
import numpy as np
from scipy.optimize import minimize
import scipy.stats as spst
STATE = 0; PARAMETER = 1
from logpdf_hyperpars import *
        
def opt4ini(sigma2,eta,inf_GMC,a,b,m,V,opt_id=np.ones(2,dtype=bool), jtopt=True, Nmax=100, thld=1e-3):
    
    # constant updates
    dlta=inf_GMC.model.misfit.stgp.I*inf_GMC.model.misfit.stgp.J/2;
    alpha = a+dlta;
    
    # initialization
    objf=np.empty(2)*np.nan;
    
    # optimization setting
    opts_unc={'gtol': 1e-6, 'disp': False, 'maxiter': 100}
    
    # optimization
    print('Optimizing parameters...\n');
    prog=np.linspace(0.05,1,20);
    t0=time.time()
    for iter in np.arange(Nmax)+1:
        # record current value
        sigma2_=sigma2;
        eta_=copy.deepcopy(eta);
        objf_=copy.deepcopy(objf);
        
        # update sigma2
        if opt_id[0]:
            dltb = inf_GMC.model.misfit.cost(inf_GMC.model.x, option='quad')*sigma2
            beta = b+dltb;
            sigma2=beta/(alpha+1); # optimize
            # objf[0]=-(spst.gamma.logpdf(1/sigma2,alpha,scale=1/beta)-2*np.log(sigma2));
            objf[0]=-spst.invgamma.logpdf(sigma2,alpha,scale=beta)
            inf_GMC.model.misfit.stgp.update(C_t=inf_GMC.model.misfit.stgp.C_t.update(sigma2 = sigma2))
        
        # update eta
        if opt_id[1]:
            if not jtopt:
                logf=[]; nl_eta=np.zeros(2)
                # eta_x
                logf.append(lambda q: logpost_eta(q,inf_GMC,m[0],V[0],[0], a=a,b=b))
                res=minimize(lambda q: -logf[0](q),eta[0],method='BFGS',options=opts_unc);
                eta[0], nl_eta[0] = res.x,res.fun
                # eta_t
                logf.append(lambda q: logpost_eta(q,inf_GMC,m[1],V[1],[1], a=a,b=b))
                res=minimize(lambda q: -logf[1](q),eta[1],method='BFGS',options=opts_unc);
                eta[1], nl_eta[1] = res.x,res.fun
                objf[1] = np.sum(nl_eta)
            else:
                logF=lambda q: logpost_eta(q,inf_GMC,m,V,[0,1], a=a,b=b)
                res=minimize(lambda q: -logF(q),eta,method='BFGS',options=opts_unc);
                eta, nl_eta = res.x,res.fun
            inf_GMC.model.misfit.stgp.update(C_x=inf_GMC.model.misfit.stgp.C_x.update(l = np.exp(eta[0])),
                                             C_t=inf_GMC.model.misfit.stgp.C_t.update(l = np.exp(eta[1])))
        
        # display the progress
        if np.isin(iter,np.floor(Nmax*prog)):
            print('%.0f of max iterations completed.\n' % (100*iter/Nmax));
        
        # display current objective function
        # fprintf(['Objective function values: ',repmat('%.4f, ',1,length(objf)),'at iteration %d.\n'], objf, iter);
        # fprintf(['Objective function values: ',repmat('%.4f, ',1,length(objf)),'at iteration %d.\n'], objf, iter);
        print('sigma2_t:%.4f, eta_x=%.4f, eta_t=%.4f.\n' %(sigma2, eta[0], eta[1]))
        print('Objective function (negative loglik) values: f_sigma2=%.4f, f_eta=%.4f at iteration %d.\n'
              %(objf[0],objf[1],iter));        
        
        # break if condition satisfied
        if iter>1:
            dif = np.zeros(2)
            dif[0]=np.max(abs(sigma2_-sigma2)); dif[1]=np.max(abs(eta_-eta));
            if np.all(abs(objf_-objf)<thld) or np.all(dif<thld):
                print('Optimization breaks at iteration %d.\n' %iter);
                break;
    
    # count time
    t1=time.time();
    print('Time used: %.2f seconds.\n' %(t1-t0));
    return sigma2,eta,inf_GMC,objf