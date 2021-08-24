#!/usr/bin/env python
"""
high-dimensional Spatio-Temporal Gaussian Process
-- given spatial and temporal kernels separately, construct a joint kernel that model the temporal evolution of spatial dependence
----------------------------------------------------------------------------------------------------------------------------------
Shiwei Lan @ UIUC, 2018
-------------------------------
Created October 25, 2018
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

import os
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# self defined modules
import sys
sys.path.append( "../../" )
from util.stgp.GP import *
from util.stgp.linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class STGP(GP):
    def __init__(self,spat,temp,Lambda=None,store_eig=False,**kwargs):
        """
        Initialize the STGP class with spatial kernel C_x, temporal kernel C_t and the dynamic eigenvalues Lambda
        C_x: spatial kernel, discrete size I x I
        C_t: temporal kernel, discrete size J x J
        Lambda: dynamic eigenvalues, discrete size J x L
        store_eig: indicator to store eigen-pairs, default to be false
        opt: kernel option (0: separable; 1: Kronecker product; 2: Kronecker sum), default to be 2
        jit: jittering term, default to be 1e-6
        spdapx: use speed-up or approximation
        -----------------------------------------------------------------------
        C_xt(z,z')= sum_{l=1}^infty lambda_l(t)*lambda_l(t')*phi_l(x)*phi_l(x')
        model 0: C_z=C_x Ox C_t (+ beta * I_x Ox I_t)
        model 1: C_z=C_xt Ox C_t (+ beta * I_x Ox I_t)
        model 2: C_z=I_x Ox C_t + (beta) * C_xt Ox I_t =: C_xt O+ C_t
        """
        if type(spat) is GP:
            self.C_x=spat # spatial kernel
        else:
            self.C_x=GP(spat,store_eig=store_eig,**kwargs)
        if type(temp) is GP:
            self.C_t=temp # temporal kernel
        else:
            self.C_t=GP(temp,store_eig=store_eig or Lambda is None,**kwargs)
        self.Lambda=Lambda if Lambda is not None else self.C_t.eigf # dynamic eigenvalues
        self.parameters=kwargs # all parameters of the kernel
        self.kappa=self.parameters.get('kappa',2) # decaying rate for dynamic eigenvalues, default to be 2
        self.opt=self.parameters.get('opt',2) # joint kernel model choice, default to be 2
        self.jit=self.parameters.get('jit',1e-6) # jitter added to joint kernel, default to be 1e-6
        self.I,self.J=self.C_x.N,self.C_t.N # spatial and temporal dimensions
        self.N=self.I*self.J # joint dimension (number of total inputs per trial)
        assert self.Lambda.shape[0]==self.J, "Size of Lambda does not match time-domain dimension!"
        self.L=self.Lambda.shape[1]
        if self.L>self.I:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed the size of spatial basis!")
            self.L=self.I; self.Lambda=self.Lambda[:,:self.I]
        self.jtkers=['sep']+['kron_'+s for s in ('prod','sum')]
        assert type(self.opt) in (int,str), "Wrong option!"
        if type(self.opt)==str:
            self.opt=np.where(self.opt in self.jtkers)
        self.ker_opt=self.jtkers[self.opt] # joint kernel
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
    
    def tomat(self,**kwargs):
        """
        Get joint kernel as a matrix
        ----------------------------
        model 0: C_xt=C_x Ox C_t, C_z=C_xt+beta*I_z (default output)
        model 1: C_xt=[sum_{l=1}^L lambda_l^alpha(t_j)*lambda_l^alpha(t_j')*phi_l(x_i)*phi_l(x_i')]_{IJ x IJ}, C_z=C_xt o (1_x Ox C_t)+beta*I_z (default output)
        model 2: C_xt=[sum_{l=1}^L lambda_l^(2alpha)(t_j)*delta(j=j')*phi_l(x_i)*phi_l(x_i')]_{IJ x IJ} (default output), C_z=(I_x Ox C_t)+beta*C_xt
        """
        alpha=kwargs.get('alpha',1) # power of dynamic eigenvalues
        beta=kwargs.get('beta',int(self.opt==2)) # coefficient before likelihood component
        trtdeg=kwargs.get('trtdeg',False) # treatment of degeneracy
        out=kwargs.get('out',{'sep':'z','kron_prod':'z','kron_sum':'xt'}[self.ker_opt]) # output option
        if self.ker_opt=='sep':
            C_xt=np.kron(self.C_t.tomat(),self.C_x.tomat()) # (IJ,IJ)
            if 'z' in out: C_z=C_xt+beta*sps.eye(self.N) # (IJ,IJ)
        else:
            Lambda_=self.Lambda**self.opt;
            if alpha<0: Lambda_[Lambda_<self.jit**2]+=self.jit**2
            Lambda_=pow(Lambda_,alpha);
            _,Phi_x=self.C_x.eigs(self.L)
            if self.ker_opt=='kron_prod':
                PhiLambda=np.reshape(Phi_x[:,None,:]*Lambda_[None,:,:],(self.N,-1),order='F') # (IJ,L)
                C_xt=PhiLambda.dot(PhiLambda.T) # (IJ,IJ)
                if trtdeg and self.L<self.I:
                    C_x0=(Phi_x*self.C_x.eigv[:self.L]).dot(Phi_x.T)
                    C_xt+=np.tile(self.C_x.tomat()-C_x0,(self.J,)*2)
                if 'z' in out: C_z=C_xt*np.kron(self.C_t.tomat(),np.ones((self.I,)*2))+((alpha>=0)*self.jit+beta)*sps.eye(self.N) # (IJ,IJ)
            elif self.ker_opt=='kron_sum':
                Lambda2Phi=Lambda_[:,None,:]*Phi_x[None,:,:] # (J,I,L)
                Lambda2Phi=Lambda2Phi.dot(Phi_x.T)+((alpha>=0)*self.jit)*np.eye(self.I)[None,:,:] # (J,I,I)
                if trtdeg and self.L<self.I:
                    C_x0=(Phi_x*self.C_x.eigv[:self.L]).dot(Phi_x.T)
                    Lambda2Phi+=np.tile(self.C_x.tomat()-C_x0,(self.J,1,1))
                C_xt=sps.block_diag(Lambda2Phi,format='csr') # (IJ,IJ)
                if 'z' in out: C_z=sps.kron(self.C_t.tomat(),sps.eye(self.I))+beta*C_xt # (IJ,IJ)
        # if self.spdapx:
        #     warnings.warn('Possible memory overflow!')
        
        output=[]
        if 'xt' in out: output.append(C_xt)
        if 'z' in out:
            if type(C_z) is np.matrix:
                C_z=C_z.getA()
            output.append(C_z)
        return output[0] if len(output)==1 else output
    
    def mult(self,v,**kwargs):
        """
        Joint kernel apply on (multiply) a function (vector): C_xt*v (default for model 2), C_z*v (default for model 0 and 1)
        """
        alpha=kwargs.pop('alpha',1) # power of dynamic eigenvalues
        beta=kwargs.pop('beta',int(self.opt==2)) # coefficient before likelihood component
        out=kwargs.pop('out',{'sep':'z','kron_prod':'z','kron_sum':'xt'}[self.ker_opt]) # output option
        if not self.spdapx:
            if v.shape[0]!=self.N:
                v=v.reshape((self.N,-1),order='F')
            C_=self.tomat(alpha=alpha,beta=beta,out=out)
            if type(C_) in [list,tuple]:
                C_xtv,C_zv=[Ci.dot(v) for Ci in C_]
            else:
                if 'xt' in out: C_xtv=C_.dot(v)
                if 'z' in out: C_zv=C_.dot(v)
        else:
            if v.shape[0]!=self.I:
                v=v.reshape((self.I,self.J,-1),order='F')
            if np.ndim(v)<3: v=v[:,:,None]
            K_=v.shape[2]
            if self.ker_opt=='sep':
                C_xtv=self.C_t.mult(self.C_x.mult(v),transp=True) # (I,J,K_)
                if 'z' in out: C_zv=C_xtv+beta*v # (I,J,K_)
            else:
                Lambda_=self.Lambda**self.opt;
                if alpha<0: Lambda_[Lambda_<self.jit**2]+=self.jit**2
                Lambda_=pow(Lambda_,alpha);
                _,Phi_x=self.C_x.eigs(self.L)
#                 Phiv=np.tensordot(Phi_x.T, v, 1) # (L,J,K_)
                prun=kwargs.pop('prun',True) and self.comm and self.I>1e3 # control of parallel run
                if prun:
#                     import pydevd; pydevd.settrace()
                    try:
                        nproc=self.comm.size; rank=self.comm.rank
                        if nproc==1: raise Exception('Only one process found!')
                        Phi_loc=Phi_x[rank::nproc,:]; v_loc=v[rank::nproc,:,:]
                        Phiv=np.empty((self.L,)+v.shape[1:])
                        self.comm.Allreduce([np.tensordot(Phi_loc.T, v_loc, 1),MPI.DOUBLE],[Phiv,MPI.DOUBLE],op=MPI.SUM)
                        pidx=np.concatenate([np.arange(self.I)[i::nproc] for i in np.arange(nproc)])
                    except Exception as e:
                        if rank==0:
                            warnings.warn('Parallel run failed: '+str(e))
                            prun=False
                            pass
                if not prun:
                    Phi_loc=Phi_x; v_loc=v
                    Phiv=np.tensordot(Phi_loc.T, v_loc, 1)
                del Phi_x, v
                if self.ker_opt=='kron_prod':
                    LambdaPhiv=(Lambda_.T[:,:,None]*Phiv).swapaxes(0,1) # (J,L,K_)
                    PhiLambda=Phi_loc[:,None,:]*Lambda_[None,:,:] # (I,J,L)
                    if 'xt' in out: C_xtv_loc=PhiLambda.dot(LambdaPhiv.sum(0)) # (I,J,K_)
                    if 'z' in out: C_zv_loc=np.sum(PhiLambda[:,:,:,None]*self.C_t.mult(LambdaPhiv,prun=not prun,**kwargs)[None,:,:,:],2)+((alpha>=0)*self.jit+beta)*v_loc # (I,J,K_)
                elif self.ker_opt=='kron_sum':
                    Lambda2Phiv=Lambda_.T[:,:,None]*Phiv # (L,J,K_)
                    C_xtv_loc=np.tensordot(Phi_loc,Lambda2Phiv,1)+((alpha>=0)*self.jit)*v_loc # (I,J,K_)
                    if 'z' in out: C_zv_loc=self.C_t.mult(v_loc,transp=True,prun=not prun,**kwargs)+beta*C_xtv_loc # (I,J,K_)
                if 'xt' in out:
                    if prun:
                        C_xtv=np.empty((self.I,self.J*K_))
                        if K_>1: C_xtv_loc=C_xtv_loc.reshape((-1,C_xtv.shape[1]))
                        self.comm.Allgatherv([C_xtv_loc,MPI.DOUBLE],[C_xtv,MPI.DOUBLE])
                        C_xtv[pidx,:]=C_xtv.copy()
                        if K_>1: C_xtv=C_xtv.reshape((self.I,self.J,-1))
                    else:
                        C_xtv=C_xtv_loc
                if 'z' in out:
                    if prun:
                        C_zv=np.empty((self.I,self.J*K_))
                        if K_>1: C_zv_loc=C_zv_loc.reshape((-1,C_zv.shape[1]))
                        self.comm.Allgatherv([C_zv_loc,MPI.DOUBLE],[C_zv,MPI.DOUBLE])
                        C_zv[pidx,:]=C_zv.copy()
                        if K_>1: C_zv=C_zv.reshape((self.I,self.J,-1))
                    else:
                        C_zv=C_zv_loc
            if 'xt' in out: C_xtv=C_xtv.reshape((self.N,-1),order='F') # (IJ,K_)
            if 'z' in out: C_zv=C_zv.reshape((self.N,-1),order='F') # (IJ,K_)
        
        output=[]
        if 'xt' in out: output.append(C_xtv)
        if 'z' in out: output.append(C_zv)
        return output[0] if len(output)==1 else output
    
    def solve(self,v,**kwargs):
        """
        Joint kernel solve a function (vector): C_xt^(-1)*v (default for model 2), C_z^(-1)*v (default for model 0 and 1)
        """
        alpha=kwargs.pop('alpha',1) # power of dynamic eigenvalues
        beta=kwargs.pop('beta',int(self.opt==2)) # coefficient before likelihood component
        out=kwargs.pop('out',{'sep':'z','kron_prod':'z','kron_sum':'xt'}[self.ker_opt]) # output option
        if self.ker_opt=='sep' and beta==0:
            if v.shape[0]!=self.I:
                v=v.reshape((self.I,self.J,-1),order='F')
            invCv=self.C_t.solve(self.C_x.solve(v),transp=True) # invC_xtv
            invCv=invCv.reshape((self.N,-1),order='F')
        elif (self.ker_opt=='sep' and beta!=0) or self.ker_opt=='kron_prod':
            if v.shape[0]!=self.N:
                v=v.reshape((self.N,-1),order='F')
            if not self.spdapx:
                trtdeg=kwargs.get('trtdeg',False) # treatment of degeneracy
                invCv=spla.solve(self.tomat(alpha=alpha,beta=beta,out=out,trtdeg=trtdeg),v,assume_a='pos')
            else:
                C_zop=spsla.LinearOperator((self.N,)*2,matvec=lambda v:self.mult(v,alpha=alpha,beta=beta,out=out,**kwargs))
#                 invCv=np.transpose([spsla.cgs(C_zop,v_j.T,**kwargs)[0] for v_j in v.T]) # invC_zv
                invCv=itsol(C_zop,v,solver='cgs',comm=kwargs.pop('comm',None))
        elif self.ker_opt=='kron_sum':
            invCv=self.mult(v,alpha=-1,beta=beta,out=out,**kwargs) # invC_xtv
        return invCv
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis of the joint kernel
        C * eigf_i = eigf_i * eigv_i, i=1,...,L; C=C_xt (for model 2), or C_z (for model 0 and 1)
        """
        if L is None:
            L=self.L
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            L=min(L,self.N)
            if self.ker_opt=='sep':
                lambda_t,Phi_t=self.C_t.eigs(); lambda_x,Phi_x=self.C_x.eigs()
                beta=kwargs.pop('beta',int(self.opt==2)) # coefficient before likelihood component
                eigv=np.kron(lambda_t,lambda_x)+beta; eigf=np.kron(Phi_t,Phi_x)
                if L<=self.C_t.L*self.C_x.L:
                    eigv=eigv[:L]; eigf=eigf[:,:L]
                else:
                    warnings.warn('Requested too many eigenvalues!')
            elif self.ker_opt=='kron_prod':
                eigv,eigf=super().eigs(L=L,upd=upd,**kwargs)
            elif self.ker_opt=='kron_sum':
                eigv=self.Lambda.flatten()**2 # (LJ,1)
                _,eigf=self.C_x.eigs(self.L); eigf=sps.kron(sps.eye(self.J),eigf).tocsc() # (IJ,LJ)
                if L<=self.L*self.J:
                    eigv=eigv[:L]; eigf=eigf[:,:L]
                else:
                    warnings.warn('Requested too many eigenvalues!')
        else:
            eigv,eigf=self.eigv,self.eigf
            eigv=eigv[:L]; eigf=eigf[:,:L]
        return eigv,eigf
    
    def act(self,x,alpha=1,**kwargs):
        """
        Obtain the action of the joint kernel C^alpha
        y=C^alpha *x; C=C_xt (for model 2), or C_z (for model 0 and 1)
        """
        if self.ker_opt in ('sep','kron_prod'):
            if x.shape[0]!=self.N:
                x=x.reshape((self.N,-1),order='F')
            y=super().act(x,alpha=alpha,**kwargs) # C_z^alpha x
        elif self.ker_opt=='kron_sum':
#             y=self.mult(x,alpha=alpha,**kwargs)
            y=self.solve(x,**kwargs) if alpha==-1 else self.mult(x,alpha=alpha,**kwargs) # C_xt^alpha x
        return y
    
    def logdet(self):
        """
        log-determinant of the joint kernel: log|C|; C=C_xt (for model 2), or C_z (for model 0 and 1)
        """
        if self.ker_opt in ('sep','kron_prod'):
            ldet=super().logdet() # log |C_z|
        elif self.ker_opt=='kron_sum':
            abs_eigv=abs(self.Lambda)
            ldet=2*np.log(abs_eigv[abs_eigv>=np.finfo(np.float).eps]).sum() # log |C_xt|
        return ldet
    
    def matn0pdf(self,X,nu=1):
        """
        logpdf of centered matrix normal distribution X ~ MN(0,C,nu*I); C=C_xt (for model 2), or C_0z (for model 0 and 1)
        """
        if self.ker_opt in ('sep','kron_prod'): # for C_0z
            if X.shape[0]!=self.N:
                X=X.reshape((self.N,-1),order='F')
            # if not self.spdapx:
            #     half_ldet=-X.shape[1]*self.logdet()/2
            #     quad=X*self.solve(X)
            # else:
                # eigv,eigf=self.eigs(); rteigv=np.sqrt(abs(eigv)); rteigv[rteigv<self.jit**2]+=self.jit**2
                # half_ldet=-X.shape[1]*np.log(rteigv).sum()
                # half_quad=eigf.T.dot(X)/rteigv[:,None]
                # quad=half_quad**2
            logpdf,half_ldet=super().matn0pdf(X,nu=nu,chol=False)
        elif self.ker_opt=='kron_sum': # for C_xt
            if X.shape[0]!=self.I:
                X=X.reshape((self.I,self.J,-1),order='F')
            if np.ndim(X)<3: X=X[:,:,None]
            Lambda_=abs(self.Lambda); Lambda_[Lambda_<self.jit**2]+=self.jit**2
            half_ldet=-X.shape[2]*np.log(Lambda_).sum()
            _,Phi_x=self.C_x.eigs(self.L)
            half_quad=np.tensordot(Phi_x.T, X, 1)/Lambda_.T[:,:,None] # (L,J,K_)
            quad=half_quad**2
            quad=-0.5*np.sum(quad)/nu
            logpdf=half_ldet+quad
        return logpdf,half_ldet
    
    def update(self,C_x=None,C_t=None,Lambda=None):
        """
        Update the eigen-basis
        """
        if C_x is not None:
            self.C_x=C_x; self.I=self.C_x.N; self.N=self.I*self.J
        if C_t is not None:
            self.C_t=C_t; self.J=self.C_t.N; self.N=self.I*self.J
        if Lambda is not None:
            assert Lambda.shape[0]==self.J, "Size of Lambda does not match time-domain dimension!"
            self.Lambda=Lambda; self.L=self.Lambda.shape[1]
            if self.L>self.I:
                warnings.warn("Karhunen-Loeve truncation number cannot exceed the size of spatial basis!")
                self.L=self.I; self.Lambda=self.Lambda[:,:self.I]
        if self.store_eig:
            self.eigv,self.eigf=self.eigs(upd=True)
        return self
    
    def scale_Lambda(self,Lambda=None,opt='up'):
        """
        Scale Lambda with the decaying rate
        u=lambda * gamma^(-alpha), gamma_l=l^(-kappa/2)
        """
        if Lambda is None:
            Lambda=self.Lambda; L=self.L
        else:
            L=Lambda.shape[1]
        if op in ('u','up'):
            alpha=1
        elif op in ('d','dn','down'):
            alpha=-1
        else:
            alpha=0
        try:
            gamma=pow(np.arange(1,L),-self.kappa/2)
        except (TypeError,ValueError):
            if 'eigCx' in self.kappa:
                gamma,_=self.C_x.eigs(L)
                gamma=np.sqrt(abs(gamma))
            else:
                gamma=np.arange(1,L)
        U=Lambda/pow(gamma,alpha)
        return U
    
    def sample_priM(self,n=1,MU=None):
        """
        Sample prior mean function (matrix normal): vec(M) ~ N(MU,C_m), C_m=C_z (for model 0/1), I_x Ox C_t (for model 2)
        Z~N(0,I_z), M=C_z^(0.5) * Z (for model 0 and 1), (C_t^(0.5) Ox I_x) * Z (for model 2)
        """
        mvn0Irv=np.random.randn(self.I,self.J,n) # (I,J)
        if self.ker_opt in ('sep','kron_prod'):
            M=self.act(mvn0Irv,alpha=0.5).reshape((self.I,self.J,-1),order='F')
        elif self.ker_opt=='kron_sum':
            M=self.C_t.act(mvn0Irv,alpha=0.5,transp=True)
        if n==1: M=np.squeeze(M,axis=2)
        if MU is not None: M+=MU # (I,J)
        return M

if __name__=='__main__':
    np.random.seed(2019)
    
    import time
    t0=time.time()
    
    # define spatial and temporal kernels
    x=np.random.randn(104,2)
    t=np.random.rand(100)
    L=100
    C_x=GP(x,L=L,store_eig=True,ker_opt='matern')
    C_t=GP(t,L=L,store_eig=True,ker_opt='powexp')
#     Lambda=C_t.rnd(n=10)
    Lambda=matnrnd(U=C_t.tomat(),V=np.eye(L))
    ker_opt=0
    stgp=STGP(C_x,C_t,Lambda,store_eig=True,opt=ker_opt)
    verbose=stgp.comm.rank==0 if stgp.comm is not None else True
    if verbose:
        print('Testing generalized STGP with %s kernel.' %(stgp.ker_opt,))
        print('Eigenvalues :', np.round(stgp.eigv[:min(10,stgp.L)],4))
        print('Eigenvectors :', np.round(stgp.eigf[:,:min(10,stgp.L)],4))
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    v=stgp.sample_priM(n=2)
    C=stgp.tomat()
    Cv=C.dot(v.reshape((stgp.N,-1),order='F'))
#     Cv_te=stgp.act(v)
    Cv_te=stgp.mult(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))
    # nonzero error with 2 cores but no error with 4 cores, maybe due to rounding error
    
    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))
    
    n=4
    v=stgp.sample_priM(n).reshape((-1,n),order='F')
#     invCv=spsla.spsolve(C,v)
#     C_op=spsla.LinearOperator((stgp.N,)*2,matvec=lambda v:stgp.mult(v))
#     invCv=itsol(C_op,v,solver='cgs')
    invCv_te=stgp.solve(v)
#     invCv_te=stgp.act(v,alpha=-1,comm=MPI.COMM_WORLD)
#     if verbose:
#         print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))
    # error decreases with L increasing for model 2 (expected)
    
    n=10
    X=stgp.sample_priM(n)
    logpdf,_=stgp.matn0pdf(X)
    if verbose:
        print('Log-pdf of a matrix normal random variable: {:.4f}'.format(logpdf))
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))
    
    u=stgp.sample_priM()[:,:,None]
    v=stgp.sample_priM()[:,:,None]
    h=1e-8
    dlogpdfv_fd=(stgp.matn0pdf(u+h*v)[0]-stgp.matn0pdf(u)[0])/h
    dlogpdfv=-stgp.solve(u).T.dot(v.flatten(order='F'))
    rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v)
    if verbose:
        print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
    if verbose:
        print('time: %.5f'% (time.time()-t3))