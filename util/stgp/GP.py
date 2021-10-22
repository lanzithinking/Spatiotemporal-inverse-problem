#!/usr/bin/env python
"""
generic Gaussian Process
-- with kernel choices 'powered exponential' and 'Matern class'
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ UIUC, 2018
-------------------------------
Created October 26, 2018
-------------------------------
Modified August 15, 2021 @ ASU
-------------------------------
https://bitbucket.org/lanzithinking/tesd_egwas
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2019, TESD project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.8"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import scipy.spatial.distance as spsd
# self defined modules
import sys
sys.path.append( "../../" )
# from __init__ import *
from util.stgp.linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class GP:
    def __init__(self,x,L=None,store_eig=False,**kwargs):
        """
        Initialize the GP class with inputs and kernel settings
        x: inputs
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        ker_opt: kernel option, default to be 'powered exponential'
        dist_f: distance function, default to be 'minkowski'
        sigma2: magnitude, default to be 1
        l: correlation length, default to be 0.5
        s: smoothness, default to be 2
        nu: matern class order, default to be 0.5
        jit: jittering term, default to be 1e-6
        # (dist_f,s)=('mahalanobis',vec) for anisotropic kernel
        spdapx: use speed-up or approximation
        """
        self.x=x # inputs
        if self.x.ndim==1: self.x=self.x[:,None]
        self.parameters=kwargs # all parameters of the kernel
        self.ker_opt=self.parameters.get('ker_opt','powexp') # kernel option
        self.dist_f=self.parameters.get('dist_f','minkowski') # distance function
        self.sigma2=self.parameters.get('sigma2',1) # magnitude
        self.l=self.parameters.get('l',0.5) # correlation length
        self.s=self.parameters.get('s',2) # smoothness
        self.nu=self.parameters.get('nu',0.5) # matern class order
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
    
    def _powexp(self,*args):
        """
        Powered exponential kernel: C(x,y)=sigma2*exp(-.5*(||x-y||/l)^s)
        """
        if len(args)==1:
            C=spsd.squareform(np.exp(-.5*pow(spsd.pdist(args[0],self.dist_f,p=self.s)/self.l,self.s)))+(1.+self.jit)*sps.eye(self.N)
        elif len(args)==2:
            C=np.exp(-.5*pow(spsd.cdist(args[0],args[1],self.dist_f,p=self.s)/self.l,self.s))
        else:
            print('Wrong number of inputs!')
            raise
        C*=self.sigma2
        return C
    
    def _matern(self,*args):
        """
        Matern class kernel: C(x,y)=2^(1-nu)/Gamma(nu)*(sqrt(2*nu)*(||x-y||/l)^s)^nu*K_nu(sqrt(2*nu)*(||x-y||/l)^s)
        """
        if len(args)==1:
            scal_dist=np.sqrt(2.*self.nu)*pow(spsd.pdist(args[0],self.dist_f,p=self.s)/self.l,self.s)
            C=pow(2.,1-self.nu)/sp.special.gamma(self.nu)*spsd.squareform(pow(scal_dist,self.nu)*sp.special.kv(self.nu,scal_dist))+(1.+self.jit)*sps.eye(self.N)
        elif len(args)==2:
            scal_dist=np.sqrt(2*self.nu)*pow(spsd.cdist(args[0],args[1],self.dist_f,p=self.s)/self.l,self.s)
            C=pow(2.,1-self.nu)/sp.special.gamma(self.nu)*pow(scal_dist,self.nu)*sp.special.kv(self.nu,scal_dist)
            C[scal_dist==0]=1
        else:
            print('Wrong number of inputs!')
            raise
        C*=self.sigma2
        return C
    
    def tomat(self):
        """
        Get kernel as matrix
        """
        kerf=getattr(self,'_'+self.ker_opt) # obtain kernel function
        C=kerf(self.x)
        if type(C) is np.matrix:
            C=C.getA()
        if self.spdapx and not sps.issparse(C):
            warnings.warn('Possible memory overflow!')
        return C
    
    def mult(self,v,**kwargs):
        """
        Kernel multiply a function (vector): C*v
        """
        transp=kwargs.get('transp',False) # whether to transpose the result
        if not self.spdapx:
            Cv=multf(self.tomat(),v,transp)
        else:
            kerf=getattr(self,'_'+self.ker_opt) # obtain kernel function
            prun=kwargs.get('prun',True) and self.comm # control of parallel run
            if prun:
                try:
#                     import pydevd; pydevd.settrace()
                    nproc=self.comm.size; rank=self.comm.rank
                    if nproc==1: raise Exception('Only one process found!')
#                     Cv_loc=self._rowmult(self.x[rank::nproc,:],v,transp)
                    Cv_loc=multf(kerf(self.x[rank::nproc,:],self.x),v,transp)
                    Cv=np.empty_like(v)
                    self.comm.Allgatherv([Cv_loc,MPI.DOUBLE],[Cv,MPI.DOUBLE])
                    pidx=np.concatenate([np.arange(self.N)[i::nproc] for i in np.arange(nproc)])
                    Cv[pidx]=Cv.copy()
                except Exception as e:
                    if rank==0:
                        warnings.warn('Parallel run failed: '+str(e))
                    prun=False
                    pass
            if not prun:
#                 Cv=np.squeeze([multf(kerf(x_i[np.newaxis,:],self.x),v,transp) for x_i in self.x],1+transp)
                Cv=np.concatenate([multf(kerf(x_i[np.newaxis,:],self.x),v,transp) for x_i in self.x])
            if transp: Cv=Cv.swapaxes(0,1)
            Cv+=self.sigma2*self.jit*v
        return Cv
    
    def solve(self,v,**kwargs):
        """
        Kernel solve a function (vector): C^(-1)*v
        """
        transp=kwargs.pop('transp',False)
        if not self.spdapx:
            invCv=mdivf(self.tomat(),v,transp)
        else:
            C_op=spsla.LinearOperator((self.N,)*2,matvec=lambda v:self.mult(v,transp=transp,prun=True))
            nd_v=np.ndim(v)
            v=v.reshape(v.shape+(1,)*(3-nd_v),order='F')
            invCv=np.array([itsol(C_op,v[:,:,k],solver='cgs',transp=transp,comm=kwargs.pop('comm',None)) for k in np.arange(v.shape[2])])
            if nd_v==3:
                invCv=invCv.transpose((1,2,0))
            else:
                invCv=np.squeeze(invCv,axis=tuple(np.arange(0,nd_v-3,-1)))
#             if transp: Cv=Cv.swapaxes(0,1)
        return invCv
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis
        C * eigf_i = eigf_i * eigv_i, i=1,...,L
        """
        if L is None:
            L=self.L;
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            maxiter=kwargs.pop('maxiter',100)
            tol=kwargs.pop('tol',1e-10)
            C_op=self.tomat() if not self.spdapx else spsla.LinearOperator((self.N,)*2,matvec=lambda v:self.mult(v,**kwargs))
            try:
                eigv,eigf=spsla.eigsh(C_op,min(L,C_op.shape[0]),maxiter=maxiter,tol=tol)
            except Exception as divg:
                print(*divg.args)
                eigv,eigf=divg.eigenvalues,divg.eigenvectors
            eigv=abs(eigv[::-1]); eigf=eigf[:,::-1]
            eigv=np.pad(eigv,(0,L-len(eigv)),mode='constant'); eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
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
    
    def matn0pdf(self,X,nu=1,chol=True):
        """
        Compute logpdf of centered matrix normal distribution X ~ MN(0,C,nu*I)
        """
        if not self.spdapx:
            if chol:
                try:
                    cholC,lower=spla.cho_factor(self.tomat())
                    half_ldet=-X.shape[1]*np.sum(np.log(np.diag(cholC)))
                    quad=X*spla.cho_solve((cholC,lower),X)
                except Exception as e:#spla.LinAlgError:
                    warnings.warn('Cholesky decomposition failed: '+str(e))
                    chol=False
                    pass
            if not chol:
                half_ldet=-X.shape[1]*self.logdet()/2
                quad=X*self.solve(X)
        else:
            eigv,eigf=self.eigs(); rteigv=np.sqrt(abs(eigv)+self.jit)#; rteigv[rteigv<self.jit**2]+=self.jit**2
            half_ldet=-X.shape[1]*np.sum(np.log(rteigv))
            half_quad=eigf.T.dot(X)/rteigv[:,None]
            quad=half_quad**2
        quad=-0.5*np.sum(quad)/nu
        logpdf=half_ldet+quad
        return logpdf,half_ldet
    
    def update(self,sigma2=None,l=None):
        """
        Update the eigen-basis
        """
        if sigma2 is not None:
            sigma2_=self.sigma2
            self.sigma2=sigma2
            if self.store_eig:
                self.eigv*=self.sigma2/sigma2_
        if l is not None:
            self.l=l
            if self.store_eig:
                self.eigv,self.eigf=self.eigs(upd=True)
        return self
    
    def rnd(self,n=1,MU=None):
        """
        Generate Gaussian random function (vector) rv ~ N(MU, C)
        """
        mvn0Irv=np.random.randn(self.N,n) # (N,n)
        rv=self.act(mvn0Irv,alpha=0.5)
        if MU is not None:
            rv+=MU
        return rv

if __name__=='__main__':
    np.random.seed(2017)
    
    import time
    t0=time.time()
    
    #     x=np.linspace(0,2*np.pi)[:,np.newaxis]
    x=np.random.randn(2000,2)
    gp=GP(x,L=10,store_eig=True,ker_opt='matern')
    verbose=gp.comm.rank==0 if gp.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(gp.eigv[:min(10,gp.L)],4))
        print('Eigenvectors :', np.round(gp.eigf[:,:min(10,gp.L)],4))
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    v=gp.rnd(n=2)
    C=gp.tomat()
    Cv=C.dot(v)
    Cv_te=gp.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))
    
    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))
    
    v=gp.rnd(n=2)
    invCv=np.linalg.solve(C,v)
#     C_op=spsla.LinearOperator((gp.N,)*2,matvec=lambda v:gp.mult(v))
#     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
    invCv_te=gp.act(v,-1)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))
    
#     X=gp.rnd(n=10)
#     X=X.reshape((X.shape[0],5,2),order='F')
#     logpdf,_=gp.matn0pdf(X)
#     if verbose:
#         print('Log-pdf of a matrix normal random variable: {:.4f}'.format(logpdf))
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))
    
    u=gp.rnd()
    v=gp.rnd()
    h=1e-5
    dlogpdfv_fd=(gp.matn0pdf(u+h*v)[0]-gp.matn0pdf(u)[0])/h
    dlogpdfv=-gp.solve(u).T.dot(v)
    rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v)
    if verbose:
        print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
    if verbose:
        print('time: %.5f'% (time.time()-t3))