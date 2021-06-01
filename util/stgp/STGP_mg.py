#!/usr/bin/env python
"""
high-dimensional Spatio-Temporal Gaussian Process (marginal)
-- given the joint stgp kernel, construct the marginal kernel in the STGP model
-------------------------------------------------------------------------------
Shiwei Lan @ UIUC, 2018
-------------------------------
Created October 25, 2018
-------------------------------
Modified October 11, 2019 @ ASU
-------------------------------
https://bitbucket.org/lanzithinking/tesd_egwas
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2019, TESD project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.6"
__maintainer__ = "Shiwei Lan"
__email__ = "shiwei@illinois.edu; lanzithinking@gmail.com; slan@asu.edu"

import os
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# self defined modules
import sys
sys.path.append( "../" )
from util.STGP import *
from util.STGP_isub import *
from util.linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class STGP_mg(STGP_isub,STGP,GP):
    def __init__(self,stgp,L=None,store_eig=False,**kwargs):
        """
        Initialize the marginal STGP class with STGP kernel stgp
        stgp: STGP kernel
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        K: number of trials
        nz_var: noise variance
        ------------------------------------------------------------------
        model 0/1: C^*=C_z + beta * I_z
                   C_z=C_x Ox C_t (for model 0), C_x|t Ox C_t (for model 1)
                   beta=nz_var/K
        model 2: C^*=I_x Ox c_t + beta * C_xt Ox I_x
                 beta=1/K
        """
        if type(stgp) is STGP:
            stgp=(stgp.C_x,stgp.C_t,stgp.Lambda) # STGP kernel
        STGP.__init__(self,spat=stgp[0],temp=stgp[1],Lambda=stgp[2],store_eig=False,**kwargs) # STGP
        self.parameters=kwargs # all parameters of the kernel
        self.K=self.parameters.get('K',1) # number of trials
        self.nz_var=self.parameters.get('nz_var',1.0) # noise variance
        if L is not None: self.L=L
        if self.L>self.I:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed the size of spatial basis!")
            self.L=self.I; self.Lambda=self.Lambda[:,:self.I]
        self.store_eig=store_eig
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self.eigs(**kwargs)
        if self.ker_opt=='kron_sum':
            self.D=self.L*self.J
    
    def tomat(self,**kwargs):
        """
        Get marginal kernel as a matrix
        -------------------------------
        model 0: C^*=(C_t Ox C_x) + nz_var/K* I_z
        model 1: C^*=C_z + nz_var/K* I_z
        model 2: C^*=(C_t Ox I_x) + 1/K* C_xt
        """
        kwargs.update(beta=self.nz_var**(self.opt!=2)/self.K,out='z')
        mgC=STGP.tomat(self,**kwargs) # STGP
        if self.N>1e3:
            warnings.warn('Possible memory overflow!')
        return mgC
    
    def mult(self,v,**kwargs):
        """
        Marginal kernel apply on (multiply) a function (vector): C_ *v, C_ as in 'tomat'
        """
        kwargs.update(beta=self.nz_var**(self.opt!=2)/self.K,out='z')
        mgCv=STGP.mult(self,v,**kwargs) # STGP
        return mgCv
    
    def solve(self,v,woodbury=True,**kwargs):
        """
        Marginal kernel solve a function (vector): C_^(-1) *v, C_ as in 'tomat'
        """
        if self.N<=1e3:
            if v.shape[0]!=self.N:
                v=v.reshape((self.N,-1),order='F')
            mgC=self.tomat()
            invmgCv=spsla.spsolve(mgC,v) if sps.issparse(mgC) else spla.solve(mgC,v,assume_a='pos')
        else:
            if self.opt==2 and woodbury:
                invmgCv=self.act(v,alpha=-1,delta=1,**kwargs) # (IJ,K_)
            else:
                if v.shape[0]!=self.N:
                    v=v.reshape((self.N,-1),order='F') # (IJ,K_)
                invmgCv=GP.solve(self,v,**kwargs) # GP
        return invmgCv
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis of the marginal kernel
        C_ * eigf_i = eigf_i * eigv_i, i=1,...,L; C_ as in 'tomat'
        """
        eigv,eigf=GP.eigs(self,L=L,upd=upd,**kwargs) # GP
        return eigv,eigf
    
    def act(self,x,alpha=1,**kwargs):
        """
        Obtain the action of the marginal kernel mgC^alpha
        y=(C^*)^alpha *x
        """
        delta=kwargs.pop('delta',None) # power of dynamic eigenvalues and C_t
        if self.opt==2 and delta is not None: # only for model 2 with computational advantage using matrix formulae
            if x.shape[0]!=self.I:
                x=x.reshape((self.I,self.J,-1),order='F') # (I,J,K_)
            if np.ndim(x)<3: x=x[:,:,None]
            K_=x.shape[2]
            _,Phi_x=self.C_x.eigs(self.L)
#             Phix=np.tensordot(Phi_x.T,x,1) # (L,J,K_)
            prun=kwargs.pop('prun',True) and self.comm and self.I>1e3
            if prun:
#                 import pydevd; pydevd.settrace()
                try:
                    nproc=self.comm.size; rank=self.comm.rank
                    if nproc==1: raise Exception('Only one process found!')
                    Phi_loc=Phi_x[rank::nproc,:]; x_loc=x[rank::nproc,:,:]
                    Phix=np.empty((self.L,)+x.shape[1:])
                    self.comm.Allreduce([np.tensordot(Phi_loc.T, x_loc, 1),MPI.DOUBLE],[Phix,MPI.DOUBLE],op=MPI.SUM)
                    pidx=np.concatenate([np.arange(self.I)[i::nproc] for i in np.arange(nproc)])
                except Exception as e:
                    if rank==0:
                        warnings.warn('Parallel run failed: '+str(e))
                        prun=False
                        pass
            if not prun:
                Phi_loc=Phi_x; x_loc=x
                Phix=np.tensordot(Phi_loc.T, x_loc, 1)
            del Phi_x, x
#             PhiSPhix=np.tensordot(Phi_loc,self.isub.act(Phix,alpha=alpha,delta=delta).reshape((self.L,self.J,-1),order='F'),1) # (I,J,K_)
#             projx=x_loc-np.tensordot(Phi_loc,Phix,1) # (I,J,K_)
#             y_loc=self.C_t.act(projx,alpha=alpha*delta,transp=True,prun=not prun,**kwargs) + PhiSPhix # (I,J,K_)
            y_loc=self.C_t.act(x_loc-np.tensordot(Phi_loc,Phix,1),alpha=alpha*delta,transp=True,prun=not prun,**kwargs) + np.tensordot(Phi_loc,super().act(Phix,alpha=alpha,delta=delta,**kwargs).reshape((self.L,self.J,-1),order='F'),1)
            if prun:
                y=np.empty((self.I,self.J*K_))
                if K_>1: y_loc=y_loc.reshape((-1,y.shape[1]))
                self.comm.Allgatherv([y_loc,MPI.DOUBLE],[y,MPI.DOUBLE])
                y[pidx,:]=y.copy()
                if K_>1: y=y.reshape((self.I,self.J,-1))
            else:
                y=y_loc
            y=y.reshape((self.N,-1),order='F') # (IJ,K_)
        else:
            y=STGP.act(self,x,alpha=alpha,**kwargs) # STGP
        return y
    
    def logdet(self,matdet=True):
        """
        log-determinant of the marginal kernel: log|C^*|
        """
        if self.opt==2 and matdet:
            ldet=super().logdet() + (self.I-self.L)*self.C_t.logdet()
        else:
            ldet=GP.logdet(self) # GP
        return ldet
    
    def matn0pdf(self,X,nu=1):
        """
        logpdf of centered matrix normal distribution X ~ MN(0,C^*,nu*I)
        """
        if X.shape[0]!=self.N:
            X=X.reshape((self.N,-1),order='F')
        if self.ker_opt in ('sep','kron_prod'):
            logpdf,half_ldet=STGP.matn0pdf(self,X=X,nu=nu) # STGP
        elif self.ker_opt=='kron_sum':
            logpdf,half_ldet=GP.matn0pdf(self,X=X,nu=nu) # GP
        return logpdf,half_ldet
    
    def update(self,stgp=None,nz_var=None):
        """
        Update the eigen-basis
        """
        if stgp is not None:
            self=STGP.update(self,C_x=stgp[0],C_t=stgp[1],Lambda=stgp[2]) # STGP
        if nz_var is not None and self.opt==1:
            nz_var_=self.nz_var
            self.nz_var=nz_var
            if self.store_eig:
                self.eigv+=(-nz_var+self.nz_var)/self.K
        return self
    
    def sample_postM(self,y,n=1,**kwargs):
        """
        Sample posterior mean function (matrix normal) vec(M) ~ N(M', C')
        See appendix B of https://arxiv.org/abs/1901.04030 for M' and C'
        """
        if np.ndim(y)==3: y=np.mean(y,2); # (I,J)
        mvn0Irv=np.random.randn(self.I,self.J,n) # (I,J)
        if self.ker_opt in ('sep','kron_prod'):
            MU=STGP.mult(self,self.solve(y),out='z') # (IJ,1) # STGP
            M=STGP.act(self,self.act(mvn0Irv*np.sqrt(self.nz_var/self.K),alpha=-0.5),alpha=0.5) # (IJ,1) # STGP
        elif self.ker_opt=='kron_sum':
            MU=self.C_t.mult(self.solve(y).reshape((self.I,self.J,-1),order='F'),transp=True) # (I,J)
            chol=(self.N<=1e3)
            if chol:
#                 C_tI_x=sps.kron(self.C_t.tomat(),sps.eye(self.I)) # (IJ,IJ)
#                 Sigma=C_tI_x.dot(self.solve(STGP.tomat(self,out='xt')/self.K)) # (IJ,IJ)
                Sigma=sps.kron(self.C_t.tomat(),sps.eye(self.I)).dot(self.solve(STGP.tomat(self,out='xt')/self.K))
                try:
#                     cholSigma=spla.cholesky(Sigma,lower=True)
#                     M=np.tensordot(cholSigma,mvn0Irv.reshape((self.N,-1),order='F'),1) # (IJ,1)
                    cholSigma,pivot=sparse_cholesky(Sigma,diag_pivot_thresh=self.jit)
                    M=pivot.dot(cholSigma).dot(mvn0Irv.reshape((self.N,-1),order='F')) # (IJ,1)
                except spla.LinAlgError:
                    warnings.warn('Cholesky decomposition failed.')
                    chol=False
                    pass
            if not chol:
                M=self.act(mvn0Irv,alpha=-0.5,delta=-1,upd=True)
            M=M.reshape((self.I,self.J,-1),order='F')
        M+=MU
        if M.shape[0]!=self.I: M=M.reshape((self.I,self.J,-1),order='F') # (I,J)
        if n==1: M=np.squeeze(M,axis=2)
        return M
    
    def predM(self,y,C_E,C_ED):
        """
        Predict the mean function based on block (cross) covariances of new points
        C_E=C_m(z_*,z_*); C_ED=C_m(z_*,Z)
        """
        if np.ndim(y)==3: y=np.mean(y,2) # (I,J)
        MU=C_ED.dot(self.solve(y))
        SIGMA=C_E-C_ED.dot(self.solve(C_ED.T))
        return MU,SIGMA

if __name__=='__main__':
    np.random.seed(2019)
    
    import time
    t0=time.time()
    
    # define spatial and temporal kernels
    x=np.random.randn(1004,2)
    t=np.random.rand(100)
    L=100
    C_x=GP(x,L=L,store_eig=True,ker_opt='matern')
    C_t=GP(t,L=L,store_eig=True,ker_opt='powexp')
#     Lambda=C_t.rnd(n=10)
    Lambda=matnrnd(U=C_t.tomat(),V=np.eye(L))
    ker_opt=2
    mg=STGP_mg((C_x,C_t,Lambda),L=L,store_eig=True,opt=ker_opt,K=2)
    verbose=mg.comm.rank==0 if mg.comm is not None else True
    if verbose:
        print('Testing marginal generalized STGP with %s kernel.' %(mg.ker_opt,))
        print('Eigenvalues :', np.round(mg.eigv[:min(10,mg.L)],4))
        print('Eigenvectors :', np.round(mg.eigf[:,:min(10,mg.L)],4))
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    v=mg.sample_priM(n=2)
    mgC=mg.tomat() # dense matrix in fact
    mgCv=mgC.dot(v.reshape((mg.N,-1),order='F'))
    mgCv_te=mg.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(mgCv-mgCv_te)/spla.norm(mgCv)))
    
    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))
    
    n=4
    v=mg.sample_priM(n).reshape((-1,n),order='F')
#     invmgCv=spsla.spsolve(mgC,v)
#     mgC_op=spsla.LinearOperator((mg.N,)*2,matvec=lambda v:mg.mult(v))
#     invmgCv=itsol(mgC_op,v,solver='cgs')
    invmgCv_te=mg.solve(v)
#     invmgCv_te=mg.act(v,alpha=-1)
#     if verbose:
#         print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invmgCv-invmgCv_te)/spla.norm(invmgCv)))
    # error decreases with L increasing for model 2 (expected)
    
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))
    
    if verbose:
        print('Log-determinant of the marginal kernel {:4f}'.format(mg.logdet()))
    
    y=v.reshape((mg.I,mg.J,-1),order='F')
    M_post=mg.sample_postM(y, n=2)
    
    if verbose:
        print('time: %.5f'% (time.time()-t3))
    