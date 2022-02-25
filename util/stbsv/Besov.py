#!/usr/bin/env python
"""
generic Besov measure
-- with basis choices 'wavelet' and 'Fourier'
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ ASU, 2022
-------------------------------
Created February 10, 2022 @ ASU
-------------------------------
https://github.com/lanzithinking/Spatiotemporal-inverse-problem
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2021, B-STIP project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# self defined modules
import sys
sys.path.append( "../../" )
# from __init__ import *
from util.stbsv.linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class Besov:
    def __init__(self,x,L=None,store_eig=False,**kwargs):
        """
        Initialize the Besov class with inputs and kernel settings
        x: inputs
        basis_opt: basis option, default to be 'Fourier'
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        sigma: magnitude, default to be 1
        l: correlation length, default to be 0.5
        s: smoothness, default to be 2
        q: norm power, default to be 1
        jit: jittering term, default to be 1e-6
        spdapx: use speed-up or approximation
        """
        self.x=x # inputs
        if self.x.ndim==1: self.x=self.x[:,None]
        self.parameters=kwargs # all parameters of the kernel
        self.basis_opt=self.parameters.get('basis_opt','Fourier') # basis option
        self.sigma=self.parameters.get('sigma',1) # magnitude
        self.l=self.parameters.get('l',0.5) # correlation length
        self.s=self.parameters.get('s',2) # smoothness
        self.q=self.parameters.get('q',1) # norm power
        self.jit=self.parameters.get('jit',1e-6) # jitter
        self.N,self.d=self.x.shape # size and dimension
        if L is None:
            L=min(self.N,100)
        self.L=L # truncation in Karhunen-Loeve expansion
        if self.L>self.N:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed size of the discrete basis!")
            self.L=self.N
        try:
            self.comm=MPI.COMM_WORLD
        except:
            print('Parallel environment not found. It may run slowly in serial.')
            self.comm=None
        self.spdapx=self.parameters.get('spdapx',self.N>1e3)
        self.store_eig=store_eig
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self.eigs(**kwargs)
    
    def _Fourier(self, x=None, L=None):
        """
        Fourier basis
        """
        if x is None:
            x=self.x
        if L is None:
            L=self.L
        if self.d==1:
            f=np.cos(np.pi*x*np.arange(L)); f[:,0]/=np.sqrt(2) # (N,L)
        elif self.d==2:
            rtL=int(np.sqrt(L))
            f=np.cos(np.pi*x[:,[0]]*(np.arange(rtL)+0.5))[:,:,None]*np.cos(np.pi*x[:,None,[1]]*(np.arange(rtL)+0.5)) # (N,rtL,rtL)
            f=f.reshape((-1,rtL**2))
            resL=L-rtL**2
            if resL>0:
                f=np.append(f,np.cos(np.pi*x[:,[0]]*(rtL+0.5))*np.cos(np.pi*x[:,[1]]*(np.arange(min(rtL,resL))+0.5)),axis=1) # row convention (type='C')
                if resL>rtL:
                    f=np.append(f,np.cos(np.pi*x[:,[0]]*(np.arange(resL-rtL)+0.5))*np.cos(np.pi*x[:,[1]]*(rtL+0.5)),axis=1)
            f*=2/np.sqrt(self.N) # (N,L)
        else:
            raise NotImplementedError('Basis for spatial dimension larger than 2 is not implemented yet!')
        return f
    
    def _wavelet(self, x=None, L=None, d=None):
        """
        (Harr) wavelet basis
        """
        if x is None:
            x=self.x
        if L is None:
            L=self.L
        if d is None:
            d=self.d
        # phi=np.sinc
        # psi=lambda x: 2*np.sinc(2*x)-np.sinc(x) # Shannon wavelet
        phi=lambda x: 1.0*(x>=0)*(x<1)
        psi=lambda x: 1.0*(x>=0)*(x<0.5) - 1.0*(x>=0.5)*(x<1) # Harr wavelet
        psi_jk=lambda x, j, k=0: 2**(j/2) * psi(2**j * x - k)
        if d==1:
            n=int(np.log2(L))
            f=phi(x)
            for j in range(n):
                f=np.append(f,psi_jk(x, j, np.arange(2**j)),axis=-1)
            if L>2**n:
                f=np.append(f,psi_jk(x, n, np.arange(L-2**n)),axis=-1) # (N,L)
        elif d==2:
            rtL=int(np.sqrt(L))
            # rtL=np.ceil(np.sqrt(L)).astype('int')
            f=self._wavelet(x[:,[0]], rtL, d=1)[:,:,None]*self._wavelet(x[:,None,[1]], rtL, d=1) # (N,rtL,rtL)
            f=f.reshape((-1,rtL**2))
            # f=f[:,:L]
            resL=L-rtL**2
            if resL>0:
                # f=np.append(f,psi_jk(x[:,[0]],j=int(np.log2(rtL)))*self._wavelet(x[:,[1]],min(rtL,resL),d=1),axis=1) # row convention (type='C')
                f=np.append(f,psi_jk(x[:,[0]],j=int(np.log2(rtL)),k=np.arange(min(rtL,resL)))*self._wavelet(x[:,[1]],min(rtL,resL),d=1),axis=1)
                if resL>rtL:
                    # f=np.append(f,self._wavelet(x[:,[0]],resL-rtL,d=1)*psi_jk(x[:,[1]],j=int(np.log2(rtL))),axis=1)
                    f=np.append(f,self._wavelet(x[:,[0]],resL-rtL,d=1)*psi_jk(x[:,[1]],j=int(np.log2(rtL)),k=np.arange(resL-rtL)),axis=1)
            # f/=np.linalg.norm(f,axis=0)
            f/=np.sqrt(self.N)
        else:
            raise NotImplementedError('Basis for spatial dimension larger than 2 is not implemented yet!')
        return f
    
    def _qrteigv(self, L=None):
        """
        Decaying (q-root) eigenvalues
        """
        if L is None:
            L=self.L
        if self.d==1:
            gamma=(self.l+np.arange(L))**(-(self.s/self.d+1./2-1./self.q))
        elif self.d==2:
            rtL=int(np.sqrt(L))
            gamma=(self.l+(np.arange(rtL)+0.5)[:,None]**2+(np.arange(rtL)+0.5)**2)**(-(self.s/self.d+1./2-1./self.q))
            gamma=gamma.flatten()
            resL=L-rtL**2
            if resL>0:
                gamma=np.append(gamma,(self.l+(rtL+0.5)**2+(np.arange(min(rtL,resL))+0.5)**2)**(-(self.s/self.d+1./2-1./self.q)))
                if resL>rtL:
                    gamma=np.append(gamma,(self.l+(np.arange(resL-rtL)+0.5)**2+(rtL+0.5)**2)**(-(self.s/self.d+1./2-1./self.q)))
        else:
            raise NotImplementedError('Basis for spatial dimension larger than 2 is not implemented yet!')
        gamma*=self.sigma**(1/self.q)
        return gamma
    
    def tomat(self):
        """
        Get kernel as matrix
        """
        eigv,eigf = self.eigs() # obtain eigen-basis
        C = (eigf*eigv).dot(eigf.T) + self.jit*sps.eye(self.N)
        # if self.spdapx and not sps.issparse(C):
        #     warnings.warn('Possible memory overflow!')
        return C
    
    def mult(self,v,**kwargs):
        """
        Kernel multiply a function (vector): C*v
        """
        transp=kwargs.get('transp',False) # whether to transpose the result
        if not self.spdapx:
            Cv=multf(self.tomat(),v,transp)
        else:
            eigv,eigf = self.eigs() # obtain eigen-pairs
#             prun=kwargs.get('prun',True) and self.comm # control of parallel run
#             if prun:
#                 try:
# #                     import pydevd; pydevd.settrace()
#                     nproc=self.comm.size; rank=self.comm.rank
#                     if nproc==1: raise Exception('Only one process found!')
#                     Cv_loc=multf(eigf[rank::nproc,:]*eigv,multf(eigf.T,v,transp),transp)
#                     Cv=np.empty_like(v)
#                     self.comm.Allgatherv([Cv_loc,MPI.DOUBLE],[Cv,MPI.DOUBLE])
#                     pidx=np.concatenate([np.arange(self.N)[i::nproc] for i in np.arange(nproc)])
#                     Cv[pidx]=Cv.copy()
#                 except Exception as e:
#                     if rank==0:
#                         warnings.warn('Parallel run failed: '+str(e))
#                     prun=False
#                     pass
#             if not prun:
#                 Cv=np.concatenate([multf(eigf_i*eigv,multf(eigf.T,v,transp),transp) for eigf_i in eigf])
#             # if transp: Cv=Cv.swapaxes(0,1)
            Cv=multf(eigf*eigv,multf(eigf.T,v,transp),transp)
            Cv+=self.jit*v
        return Cv
    
    def solve(self,v,**kwargs):
        """
        Kernel solve a function (vector): C^(-1)*v
        """
        transp=kwargs.pop('transp',False)
        if not self.spdapx:
            invCv=mdivf(self.tomat(),v,transp)
        else:
#             C_op=spsla.LinearOperator((self.N,)*2,matvec=lambda v:self.mult(v,transp=transp,prun=True))
#             nd_v=np.ndim(v)
#             v=v.reshape(v.shape+(1,)*(3-nd_v),order='F')
#             invCv=np.array([itsol(C_op,v[:,:,k],solver='cgs',transp=transp,comm=kwargs.pop('comm',None)) for k in np.arange(v.shape[2])])
#             if nd_v==3:
#                 invCv=invCv.transpose((1,2,0))
#             else:
#                 invCv=np.squeeze(invCv,axis=tuple(np.arange(0,nd_v-3,-1)))
# #             if transp: Cv=Cv.swapaxes(0,1)
            eigv,eigf = self.eigs() # obtain eigen-pairs
            invCv=multf(eigf/(self.jit+eigv),multf(eigf.T,v,transp),transp)
        return invCv
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis
        C * eigf_i = eigf_i * eigv_i, i=1,...,L
        """
        if L is None:
            L=self.L;
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            basisf=getattr(self,'_'+self.basis_opt) # obtain basis function
            eigf=basisf(x=self.x, L=L)
            if eigf.shape[1]<L:
                eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
                warnings.warn('zero eigenvectors padded!')
            eigv=self._qrteigv(L)**self.q
            if len(eigv)<L:
                eigv=np.pad(eigv,[0,L-len(eigv)],mode='constant')
                warnings.warn('zero eigenvalues padded!')
        else:
            eigv,eigf=self.eigv,self.eigf
            eigv=eigv[:L]; eigf=eigf[:,:L]
        return eigv,eigf
    
    def act(self,x,alpha=1,**kwargs):
        """
        Obtain the action of C^alpha
        y=C^alpha *x
        """
        transp=kwargs.get('transp',False)
        if alpha==0:
            y=x
        elif alpha==1:
            y=self.mult(x,**kwargs)
        elif alpha==-1:
            y=self.solve(x,**kwargs)
        else:
            eigv,eigf=self.eigs(**kwargs)
            # if alpha<0: eigv[eigv<self.jit**2]+=self.jit**2
            y=multf(eigf*pow((alpha<0)*self.jit+eigv,alpha),multf(eigf.T,x,transp),transp)
        return y
    
    def logdet(self):
        """
        Compute log-determinant of the kernel C: log|C|
        """
        eigv,_=self.eigs()
        abs_eigv=abs(eigv)
        ldet=np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(np.float).eps]))
        return ldet
    
    def logpdf(self,X):
        """
        Compute logpdf of centered Besov distribution X ~ Besov(0,C)
        """
        if not self.spdapx:
            _,eigf = self.eigs()
            proj_X = eigf.T.dot(self.act(X, alpha=-1/self.q))
            q_ldet=-X.shape[1]*self.logdet()/self.q
        else:
            basisf=getattr(self,'_'+self.basis_opt) # obtain basis function
            eigf=basisf()
            qrt_eigv=self._qrteigv()
            q_ldet=-X.shape[1]*np.sum(np.log(qrt_eigv))
            proj_X=eigf.T.dot(X)/qrt_eigv
        qsum=-0.5*np.sum(abs(proj_X)**self.q)
        logpdf=q_ldet+qsum
        return logpdf,q_ldet
    
    def update(self,sigma=None,l=None):
        """
        Update the eigen-basis
        """
        if sigma is not None:
            sigma_=self.sigma
            self.sigma=sigma
            if self.store_eig:
                self.eigv*=self.sigma/sigma_
        if l is not None:
            self.l=l
            if self.store_eig:
                self.eigv,self.eigf=self.eigs(upd=True)
        return self
    
    def rnd(self,n=1):
        """
        Generate Besov random function (vector) rv ~ Besov(0,C)
        """
        lap_rv=np.random.laplace(size=(self.L,n)) # (L,n)
        eigv,eigf=self.eigs()
        rv=eigf.dot(np.sign(lap_rv)*(eigv[:,None]*abs(lap_rv))**(1/self.q)) # (N,n)
        # L=int(np.sqrt(self.L))**2
        # rv=eigf[:,:L].dot((np.sign(lap_rv)*(eigv[:,None]*abs(lap_rv))**(1/self.q))[:L])
        return rv

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    # x=np.random.rand(64**2,2)
    # x=np.stack([np.sort(np.random.rand(64**2)),np.sort(np.random.rand(64**2))]).T
    xx,yy=np.meshgrid(np.linspace(0,1,128),np.linspace(0,1,128))
    x=np.stack([xx.flatten(),yy.flatten()]).T
    bsv=Besov(x,L=1000,store_eig=True,basis_opt='wavelet', q=1.0) # constrast with q=2.0
#     verbose=bsv.comm.rank==0 if bsv.comm is not None else True
#     if verbose:
#         print('Eigenvalues :', np.round(bsv.eigv[:min(10,bsv.L)],4))
#         print('Eigenvectors :', np.round(bsv.eigf[:,:min(10,bsv.L)],4))
#
#     t1=time.time()
#     if verbose:
#         print('time: %.5f'% (t1-t0))
#
#     v=bsv.rnd(n=2)
#     C=bsv.tomat()
#     Cv=C.dot(v)
#     Cv_te=bsv.act(v)
#     if verbose:
#         print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))
#
#     t2=time.time()
#     if verbose:
#         print('time: %.5f'% (t2-t1))
#
#     v=bsv.rnd(n=2)
#     invCv=np.linalg.solve(C,v)
# #     C_op=spsla.LinearOperator((bsv.N,)*2,matvec=lambda v:bsv.mult(v))
# #     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
#     invCv_te=bsv.act(v,-1)
#     if verbose:
#         print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))
#
# #     X=bsv.rnd(n=10)
# #     X=X.reshape((X.shape[0],5,2),order='F')
# #     lobsvdf,_=bsv.logpdf(X)
# #     if verbose:
# #         print('Log-pdf of a matrix normal random variable: {:.4f}'.format(lobsvdf))
#     t3=time.time()
#     if verbose:
#         print('time: %.5f'% (t3-t2))
#
    # u=bsv.rnd()
#     v=bsv.rnd()
#     h=1e-5
#     dlogpdfv_fd=(bsv.logpdf(u+h*v)[0]-bsv.logpdf(u)[0])/h
#     dlogpdfv=-bsv.solve(u).T.dot(v)
#     rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v)
#     if verbose:
#         print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
#     if verbose:
#         print('time: %.5f'% (time.time()-t3))
    
    import matplotlib.pyplot as plt
    
    fig, axes=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True,figsize=(15,12))
    for i,ax in enumerate(axes.flat):
        ax.imshow(bsv.eigf[:,i].reshape((int(np.sqrt(bsv.eigf.shape[0])),-1)),origin='lower')
        ax.set_aspect('auto')
    plt.show()
    
    u=bsv.rnd(n=25)
    fig, axes=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True,figsize=(15,12))
    for i,ax in enumerate(axes.flat):
        ax.imshow(u[:,i].reshape((int(np.sqrt(u.shape[0])),-1)),origin='lower')
        ax.set_aspect('auto')
    plt.show()