#!/usr/bin/env python
"""
high-dimensional Spatio-Temporal Gaussian Process (marginal in subspace)
-- given the joint stgp kernel, construct the marginal kernel in an intrinsic subspace in the STGP model
-- mainly used in gSTGP model II with computational advantage using matrix formulae
--------------------------------------------------------------------------------------------------------
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
# from util.GP import *
from util.STGP import *
from util.linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class STGP_isub(STGP):#,GP):
    def __init__(self,stgp,L=None,store_eig=False,**kwargs):
        """
        Initialize the subspace marginal STGP class with STGP kernel stgp
        stgp: STGP kernel
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        K: number of trials
        ---------------------
        C_isub:=C_t^delta Ox I_L + K^(-delta) diag( vec^T(Lambda^(2delta)) )
        delta=1: the above kernel appears in the inverse of marginal kernel invC^* of model II
        delta=-1: the above kernel appears in the posterior covariance of mean function C' of model II
        refer to appendix B.3 of https://arxiv.org/abs/1901.04030
        """
        if type(stgp) is STGP:
            stgp=(stgp.C_x,stgp.C_t,stgp.Lambda) # STGP kernel
        super().__init__(spat=stgp[0],temp=stgp[1],Lambda=stgp[2],store_eig=False,**kwargs) # STGP
        self.parameters=kwargs # all parameters of the kernel
        self.K=self.parameters.get('K',1) # number of trials
        if L is not None: self.L=L
        if self.L>self.I:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed the size of spatial basis!")
            self.L=self.I; self.Lambda=self.Lambda[:,:self.I]
        self.D=self.L*self.J
        self.store_eig=store_eig
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self._eigs(**kwargs)
        if self.ker_opt=='kron_prod':
            warnings.warn('STGP_isub is for STGP model II (Kronecker sum)! Using it in STGP model 0 (separable product) or I (Kronecker product) yields unexpected results!')
    
    def _tomat(self,**kwargs):
        """
        Get marginal kernel in subspace as a matrix
        C_isub=C_t^delta Ox I_L + K^(-delta) diag( vec^T(Lambda^(2delta)) )
        """
        delta=kwargs.get('delta',1) # power of dynamic eigenvalues
        S=sps.diags(pow(self.Lambda**2/self.K,delta).flatten(),shape=(self.D,)*2) # (LJ,LJ)
        if delta==1:
            S+=sps.kron(self.C_t.tomat(),sps.eye(self.L)) # (LJ,LJ)
        else:
            S+=sps.kron(self.C_t.act(np.eye(self.J),alpha=delta),sps.eye(self.L)) # (LJ,LJ)
        if self.D>1e3:
            warnings.warn('Possible memory overflow!')
        return S
    
    def _mult(self,v,**kwargs):
        """
        action of the kernel in the intrinsic subspace (for model II only): C_isub *v
        """
        delta=kwargs.pop('delta',1) # power of dynamic eigenvalues and C_t
        if self.D<=1e3:
            if v.shape[0]!=self.D:
                v=v.reshape((self.D,-1),order='F') # (LJ,K_)
            Sv=self._tomat(delta=delta).dot(v) # (LJ,K_)
        else:
            if v.shape[0]!=self.L:
                v=v.reshape((self.L,self.J,-1),order='F'); # (L,J,K_)
            if np.ndim(v)<3: v=v[:,:,None]
            Sv=pow(self.Lambda.T**2/self.K,delta)[:,:,None]*v+self.C_t.act(v,alpha=delta,transp=True,**kwargs) # (L,J,K_)
            Sv=Sv.reshape((self.D,-1),order='F') # (LJ,K_)
        return Sv
    
    def _solve(self,v,**kwargs):
        """
        solving of the kernel in the intrinsic subspace (for model II only): C_isub^(-1) *v
        """
        delta=kwargs.pop('delta',1) # power of dynamic eigenvalues and C_t
        if v.shape[0]!=self.D:
            v=v.reshape((self.D,-1),order='F') # (LJ,K_)
        if delta==-1:
            lambda2=self.Lambda.flatten()**2/self.K
            lambda2v=lambda2[:,None]*v
        if self.D<=1e3:
            S_mat=self._tomat(delta=delta**(delta!=-1))
            if delta==-1:
#                 invSv=(S_mat-sps.diags(lambda2)).dot(spsla.spsolve(S_mat,lambda2[:,None]*v))
                invSv=lambda2v-lambda2[:,None]*spsla.spsolve(S_mat,lambda2v)
            else:
                invSv=spsla.spsolve(S_mat,v)
        else:
            S_op=spsla.LinearOperator((self.D,)*2,matvec=lambda v:self._mult(v,delta=delta**(delta!=-1),**kwargs))
            if delta==-1:
#                 invSv=self.C_t.mult(np.transpose([spsla.cgs(S_op,lambda2*v_j.T)[0] for v_j in v.T]).reshape((self.L,self.J,-1),order='F'),transp=True).reshape((self.D,-1),order='F') # (LJ,K_)
#                 invSv=lambda2v-lambda2[:,None]*np.transpose([spsla.cgs(S_op,v_j.T)[0] for v_j in lambda2v.T])
                invSv=itsol(S_op,lambda2v,solver='cgs',comm=kwargs.pop('comm',None))
                invSv=lambda2v-lambda2[:,None]*invSv
            else:
#                 invSv=np.transpose([spsla.cgs(S_op,v_j.T,**kwargs)[0] for v_j in v.T]) # (LJ,K_)
                invSv=itsol(S_op,v,solver='cgs',comm=kwargs.pop('comm',None))
        return invSv
    
    def _eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis of the marginal kernel in the intrinsic subspace
        C_isub * v = v* lambda
        output: lambda^(alpha), v
        """
        if L is None:
            L=self.L;
        alpha=kwargs.pop('alpha',1) # power of requested eigen-values
        delta=kwargs.pop('delta',1) # power of dynamic eigenvalues and C_t
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            L=min(L,self.D)
            maxiter=kwargs.pop('maxiter',100)
            tol=kwargs.pop('tol',1e-10)
            if self.D<=1e3:
                S_op=self._tomat(delta=delta) if alpha>0 else self._solve(np.eye(self.D),delta=delta)
            else:
                S_op=spsla.LinearOperator((self.D,)*2,matvec=lambda v:self._mult(v,delta=delta,**kwargs) if alpha>0 else self._solve(v,delta=delta,**kwargs))
            try:
                eigv,eigf=spsla.eigsh(S_op,L,maxiter=maxiter,tol=tol)
            except Exception as divg:
                print(*divg.args)
                eigv,eigf=divg.eigenvalues,divg.eigenvectors
            eigv=pow(abs(eigv[::-1]),abs(alpha)); eigf=eigf[:,::-1]
            eigv=np.pad(eigv,(0,L-len(eigv)),mode='constant'); eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
        else:
            eigv,eigf=self.eigv,self.eigf
            eigv=pow(eigv[:L],alpha); eigf=eigf[:,:L]
        return eigv,eigf
    
    def act(self,x,alpha=1,**kwargs):
        """
        Obtain the action of the marginal kernel in the intrinsic subspace: C_isub^(alpha) *v
        """
        transp=kwargs.get('transp',False)
        if alpha==0:
            y=x;
        elif alpha==1:
            y=self._mult(x,**kwargs)
        elif alpha==-1:
            y=self._solve(x,**kwargs)
        else:
            if x.shape[0]!=self.D:
                x=x.reshape((self.D,-1),order='F')
            chol=(abs(alpha)==0.5 and self.D<=1e3)
            if chol:
                try:
    #                 cholS,lower=spla.cho_factor(self._tomat(**kwargs),lower=True) # not working for sparse matrix
    #                 y=np.tril(cholS).dot(x) if alpha>0 else spla.cho_solve((cholS,lower),x)
                    cholS,pivot=sparse_cholesky(self._tomat(**kwargs),diag_pivot_thresh=self.jit)
                    y=pivot.dot(cholS.dot(x)) if alpha>0 else spsla.spsolve(pivot.dot(cholS),x)
                except Exception:#spla.LinAlgError:
                    warnings.warn('Cholesky decomposition failed.')
                    chol=False
                    pass
            if not chol:
                eigv,eigf=self._eigs(**kwargs)
                y=(eigf*pow((alpha<0)*self.jit+eigv,alpha)).dot(eigf.T.dot(x))
#         if x.shape[0]!=self.D:
#             x=x.reshape((self.D,-1),order='F')
#         y=GP.act(self,x,alpha=alpha,**kwargs)
        return y
    
    def logdet(self):
        """
        log-determinant of the marginal kernel in the intrinsic subspace: log|C_isub|
        """
        eigv,_=self._eigs()
        abs_eigv=abs(eigv)
        ldet=np.log(abs_eigv[abs_eigv>=np.finfo(np.float).eps]).sum()
#         ldet=GP.logdet(self) # GP
        return ldet
    
    def update(self,stgp=None):
        """
        Update the eigen-basis
        """
        if stgp is not None:
            super().update(C_x=stgp[0],C_t=stgp[1],Lambda=stgp[2]) # STGP
        return self

if __name__=='__main__':
    np.random.seed(2019)
    
    import time
    t0=time.time()
    
    # define spatial and temporal kernels
    x=np.random.randn(100,2)
    t=np.random.rand(11)
    L=10
    C_x=GP(x,L=L,store_eig=True,ker_opt='matern')
    C_t=GP(t,L=L,store_eig=True,ker_opt='powexp')
#     Lambda=C_t.rnd(n=10)
    Lambda=matnrnd(U=C_t.tomat(),V=np.eye(L))
    ker_opt=2
    isub=STGP_isub((C_x,C_t,Lambda),L=L,store_eig=True,opt=ker_opt,K=2)
    verbose=isub.comm.rank==0 if isub.comm is not None else True
    if verbose:
        print('Testing generalized subspace marginal STGP with %s kernel.' %(isub.ker_opt,))
        print('Eigenvalues :', np.round(isub.eigv[:min(10,isub.L)],4))
        print('Eigenvectors :', np.round(isub.eigf[:,:min(10,isub.L)],4))
    
    
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    v=matnrnd(U=np.eye(L),V=C_t.tomat())
    S=isub._tomat(delta=-1)
    Sv=S.dot(v.reshape((isub.D,-1),order='F'))
    Sv_te=isub.act(v,delta=-1)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Sv-Sv_te)/spla.norm(Sv)))
    
    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))
    
    n=4
    v=matnrnd(U=np.eye(L),V=C_t.tomat(),n=n).reshape((isub.D,-1),order='F')
    invSv=spsla.spsolve(S,v)
#     S_op=spsla.LinearOperator((isub.D,)*2,matvec=lambda v:isub.mult(v,delta=-1))
#     invSv=np.array([spsla.cgs(S_op,v[:,j])[0] for j in np.arange(n)]).T
#     invSv_te=isub._solve(v,alpha=-1,comm=MPI.COMM_WORLD)
#     invSv_te=isub._solve(v,delta=-1)
    invSv_te=isub.act(v,alpha=-1,delta=-1,comm=MPI.COMM_WORLD)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invSv-invSv_te)/spla.norm(invSv)))
    # error decreases with L increasing for model 2 (expected)
    
    if verbose:
        print('Log-determinant of the kernel in the intrinsic subspace {:4f}'.format(isub.logdet()))
    
    if verbose:
        print('time: %.5f'% (time.time()-t2))