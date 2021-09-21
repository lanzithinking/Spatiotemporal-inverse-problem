#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:22:33 2021
Slice sampling (1d) algorithm by Neal (2003)
@author: Shuyi
"""
import numpy as np

def slice(q0,l0,logf,w=1,m=5,bdy=[-float('inf'),float('inf')]):
    '''
    Slice sampling (1d) algorithm by Neal (2003)
    --------------------------------------------
    inputs:
      q0: initial state of the parameters
      l0: initial log-density
      logf: log-density function of q
      w: estimate size of a slice
      m: integer limiting th size of a slice to mw
    outputs:
      q: new state of the parameter following N(q;0,C)*lik
      l: new log-density
    '''
    rand = lambda:np.random.rand(1)[0]
    
    # log-density threshold (defines a slice)
    logy = l0 + np.log(rand());
    
    # step out to obtain the [L,R] range
    L = q0 - w*rand();
    R = L + w;
    J = np.floor(m*rand());
    K = (m-1) - J;
    
    # make sure [L,R] is within boundary
    L = max(L,bdy[0]);
    R = min(R,bdy[1]);
    
    while 1:
        if J<=0 or logy>=logf(L): break
        L = L - w;
        L = max(L,bdy[0]);
        J = J - 1;
    
    while 1:
        if K<=0 or logy>=logf(R): break;
        R = R + w;
        R = min(R,bdy[1]);
        K = K - 1;
    
    # shrink to obtain a sample
    while 1:
        q = L + rand()*(R-L);
        l = logf(q);
        if l>logy:
            break;
        
        # shrink the bracket and try a new point
        if q<q0:
            L = q;
        else:
            R = q;
    
    return q,l

    
    


