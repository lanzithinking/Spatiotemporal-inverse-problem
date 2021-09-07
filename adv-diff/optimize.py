#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 15:41:55 2021

@author: apple
"""
import time
import numpy as np
from scipy.optimize import minimize
import scipy.stats as spst
import copy


def logf(eta, inf_GMC, m, V, option=1):
    # return the log posterior = loglik + logpri
    # option=1: logposterior for x, option=2: logposterior for t
    if option==1:
        inf_GMC.model.misfit.stgp.update(C_x=inf_GMC.model.misfit.stgp.C_x.update(l = np.exp(eta)))
    elif option==2:
        inf_GMC.model.misfit.stgp.update(C_t=inf_GMC.model.misfit.stgp.C_t.update(l = np.exp(eta)))
    loglik = inf_GMC.ll
    logpri = -.5*(eta-m)**2/V
    return (loglik + logpri)
        
def opt4ini(sigma2,eta,inf_GMC,a,b,m,V,opt_id=np.ones(2,dtype=bool), Nmax=100, thld=1e-3):
    
    PARAMETER = 1
    STATE = 0
    objf=np.empty(2)*np.nan;
    # constant updates
    dlta=inf_GMC.model.misfit.stgp.I*inf_GMC.model.misfit.stgp.J/2;
    alpha = a+dlta;
    
    
    # initialization
    #stgp=STGP(ker1['C'],ker2['C'],Lambda,mdl_opt);
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>optimization setting
    #opts_unc=optimoptions('fminunc','Algorithm','quasi-newton','display','off','MaxIterations',100);
    #opts_con=optimoptions('fmincon','Algorithm','sqp','display','off','MaxIterations',100);
    
    # optimization
    print('Optimizing parameters...\n');
    prog=np.linspace(0.05,1,20);
    t0=time.time()
    #>>>>>>>>>>>>>what's meaning of cycle??????/
    for iter in np.arange(Nmax)+1:
        # record current value
        sigma2_=sigma2;
        eta_=copy.deepcopy(eta);
        objf_=copy.deepcopy(objf);
        
        # display the progress
#        if ismember(iter,floor(Nmax.*prog))
#            fprintf('%.0f%% of max iterations completed.\n',100*iter/Nmax);
        
        # update sigma2_t
        if opt_id[0]==1: 
                        
            inf_GMC.model.x[PARAMETER] = inf_GMC.q
            inf_GMC.model.pde.solveFwd(inf_GMC.model.x[STATE], inf_GMC.model.x)
            dltb = inf_GMC.model.misfit.cost(inf_GMC.model.x, option='diff')
            
            beta = b+dltb;                                   
            sigma2=beta/(alpha+1); # optimize
            objf[0]=(np.log(spst.gamma.pdf(1/sigma2,alpha,scale=1/beta))-2*np.log(sigma2));
            
            inf_GMC.model.misfit.stgp.update(C_t=inf_GMC.model.misfit.stgp.C_t.update(sigma2 = sigma2))
        
        
        # update eta
        if opt_id[1]==1: 
            # eta_x
            logf1 = lambda eta:logf(eta,inf_GMC,m[0],V[0],1)
            minuslogf1 = lambda q: -logf1(q)
            res=minimize(minuslogf1,eta[0],method='BFGS',
                           options={'gtol': 1e-6, 'disp': False, 'maxiter': 100 });
            eta[0],l_eta_x = res.x,-res.fun  
                
            # eta_t
            logf2 =lambda eta:logf(eta,inf_GMC,m[1],V[1],2)        
            minuslogf2 = lambda q: -logf2(q)
            res=minimize(minuslogf2,eta[1],method='BFGS',
                           options={'gtol': 1e-6, 'disp': False, 'maxiter': 100 });
            eta[1], l_eta_t = res.x,-res.fun
            objf[1] = l_eta_x + l_eta_t
            inf_GMC.model.misfit.stgp.update(C_x=inf_GMC.model.misfit.stgp.C_x.update(l = np.exp(eta[0])),
                                            C_t=inf_GMC.model.misfit.stgp.C_t.update(l = np.exp(eta[1])))
        
        # display the progress
        if np.isin(iter,np.floor(Nmax*prog)):
            print('%.0f of max iterations completed.\n' % (100*iter/Nmax));
                
        # display current objective function
        #  fprintf(['Objective function values: ',repmat('%.4f, ',1,length(objf)),'at iteration %d.\n'], objf, iter);
        print('sigma2_t:%.4f, eta_x=%.4f, eta_t=%.4f.\n' %(sigma2, eta[0], eta[1]))
        print('Objective function(loglik) values: f_sigma2=%.4f, f_eta_x=%.4f, f_eta_t=%.4f at iteration %d.\n'
              %(objf[0],l_eta_x,l_eta_t,iter));        
        # break if condition satisfied
        if iter>1:
            dif = np.zeros(2)
            dif[0]=np.max(abs(sigma2_-sigma2)); dif[1]=np.max(abs(eta_-eta));

            if np.all(abs(objf_-objf)<thld) or np.all(dif<thld):
                print('Optimization breaks at iteration %d.\n' %iter);
                break;       
        # display current objective function
    #     fprintf(['Objective function values: ',repmat('%.4f, ',1,length(objf)),'at iteration %d.\n'], objf, iter);
           
    # count time
    t1=time.time();
    print('Time used: %.2f seconds.\n' %(t1-t0));
    return sigma2,eta,inf_GMC,objf 