#!/usr/bin/env python
"""
Class definition of Besov prior for dynamic linear model.
---------------------------------------------------------------
Created February 15, 2022 for project of Bayesian Spatiotemporal inverse problem (B-STIP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu lanzithinking@outlook.com"

import os
import numpy as np
import scipy as sp
import scipy.sparse as sps

# self defined modules
import os,sys
sys.path.append( "../" )
from util.stbsv.Besov import Besov

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class prior(Besov):
    """
    Besov prior measure B(mu,_C) defined on 2d domain V.
    """
    def __init__(self,meshsz,L=None,mean=None,store_eig=False,**kwargs):
        if not hasattr(meshsz, "__len__"):
            self.meshsz=(meshsz,)*2
        else:
            self.meshsz=meshsz
        self._mesh()
        self.space=kwargs.pop('space','vec') # alternative 'fun'
        super().__init__(x=self.mesh, L=L, store_eig=store_eig, **kwargs)
        self.mean=mean
    
    def _mesh(self):
        """
        Build the mesh
        """
        # set the mesh
        xx,yy=np.meshgrid(np.linspace(0,1,self.meshsz[0]),np.linspace(0,1,self.meshsz[1]))
        self.mesh=np.stack([xx.flatten(),yy.flatten()]).T
        # print('\nThe mesh is defined.')
    
    def cost(self,u):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        -.5* ||_C^(-1/q) u(x)||^q = -.5 sum_l |gamma_l^{-1} <phi_l, u>|^q
        """
        if self.mean is not None:
            u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        proj_u=self.C_act(u, -1.0/self.q, proj=True)
        
        val=0.5*np.sum(abs(proj_u)**self.q)
        return val
    
    def grad(self,u):
        """
        Calculate the gradient of log-prior
        """
        if self.mean is not None:
            u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        eigv, eigf=self.eigs()
        if self.space=='vec':
            proj_u=u/eigv**(1/self.q)
            g=0.5*self.q*abs(proj_u)**(self.q-1) *np.sign(proj_u)/eigv**(1/self.q)
        elif self.space=='fun':
            proj_u=eigf.T.dot(u)/eigv**(1/self.q)
            g=eigf.dot(0.5*self.q*abs(proj_u)**(self.q-1) *np.sign(proj_u)/eigv**(1/self.q))
        else:
            raise ValueError('Wrong space!')
        return g
        
        
    def sample(self):
        """
        Sample a random function u ~ B(0,_C)
        vector u ~ B(0,K): u = gamma |xi|^(1/q), xi ~ Lap(0,1)
        """
        if self.space=='vec':
            lap_rv=np.random.laplace(size=self.L) # (L,n)
            eigv,eigf=self.eigs()
            u=np.sign(lap_rv)*(eigv*abs(lap_rv))**(1/self.q) # (L,n)
        elif self.space=='fun':
            u=super().rnd(n=1).squeeze()
            if self.mean is not None:
                u+=self.mean
            u=u.reshape(self.meshsz)
        else:
            raise ValueError('Wrong space!')
        return u
    
    
    def C_act(self,u,comp=1,proj=False):
        """
        Calculate operation of C^comp on vector u: u --> C^comp * u
        """
        if np.ndim(u)>1: u=u.flatten()
          
        if comp==0:
            return u
        else:
            eigv, eigf=self.eigs()
            if self.space=='vec':
                proj_u=u*eigv**(comp)
            elif self.space=='fun':
                proj_u=eigf.T.dot(u)*eigv**(comp)
            else:
                raise ValueError('Wrong space!')
            if proj or self.space=='vec':
                return proj_u
            else:
                Cu=eigf.dot(proj_u)
                return Cu
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(2021)
    # define the prior
    meshsz=128
    prior = prior(meshsz=meshsz, basis_opt='Fourier', q=1.0, L=400, space='fun')
    # generate sample
    u=prior.sample()
    nlogpri=prior.cost(u)
    ngradpri=prior.grad(u)
    print('The negative logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(nlogpri,np.linalg.norm(ngradpri)))
    # test
    h=1e-5
    v=prior.sample()
    ngradv_fd=(prior.cost(u+h*v)-nlogpri)/h
    ngradv=ngradpri.dot(v.flatten())
    rdiff_gradv=np.abs(ngradv_fd-ngradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    # plot
    plt.imshow(u,origin='lower',extent=[0,1,0,1])
    plt.show()
    