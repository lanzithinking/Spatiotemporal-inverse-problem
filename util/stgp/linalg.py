#!/usr/bin/env python
"""
Linear Algebra
-- some functions written for convenience in STGP models
--------------------------------------------------------
Shiwei Lan @ ASU, 2019
-------------------------------
Created November 23, 2018
-------------------------------
https://bitbucket.org/lanzithinking/tesd_egwas
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2019, TESD project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "shiwei@illinois.edu; lanzithinking@gmail.com; slan@asu.edu"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# import scipy.spatial.distance as spsd
# try:
#     from mpi4py import MPI
# except ImportError:
#     print('mpi4py not installed! It may run slowly...')
#     pass

# self defined modules
import sys
sys.path.append( "../../" )
from util.stgp.__init__ import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def multf(a,b,transp=False):
    """
    matrix multiplication function a*b with a being a (square) matrix, b being a vector, matrix or 3d array
    'transp' indicates whether b is transposed already; returning result is in the same layout as b
    """
    if np.ndim(b)<=2: # optional
        c=b.dot(a.T) if transp else a.dot(b)
    elif np.ndim(b)==3: # np.matmul(a,b)
        c=a.dot(b).swapaxes(0,1) if transp else np.tensordot(a,b,1)
    else:
        raise Exception('Wrong dimension of b!')
    return c
    
def mdivf(a,b,transp=False):
    """
    matrix division (multiply by inverse) function a*b^(-1) with a being a (square) matrix, b being a vector, matrix or 3d array
    'transp' indicates whether b is transposed already; returning result is in the same layout as b
    """
    if transp:
        if np.ndim(b)<=2: # optional
            try:
                c=spla.solve(a,b.T,assume_a='pos').T
            except spla.LinAlgError:
                c=spla.solve(a,b.T).T
        elif np.ndim(b)==3:
            try:
                c=spla.solve(a,b.swapaxes(0,1),assume_a='pos').swapaxes(0,1)
            except spla.LinAlgError:
                c=spla.solve(a,b.swapaxes(0,1)).swapaxes(0,1)
        else:
            raise Exception('Wrong dimension of b!')
    else:
        try:
            c=spla.solve(a,b,assume_a='pos')
        except spla.LinAlgError:
            c=spla.solve(a,b)
    return c
    
def itsol(a,b,solver='cg',transp=False,comm=None,**kwargs):
    """
    iterative solver for multiple rhs
    """
    nd_b=np.ndim(b)
    if nd_b==1: b=b[:,np.newaxis]
    if transp: b=b.T
    solve=getattr(spsla,solver)
    maxiter=kwargs.get('maxiter',None)
    tol=kwargs.get('tol',1e-5)
    prun=comm is not None and nd_b>1
    if prun:
        try:
#             import pydevd; pydevd.settrace()
            b_loc=np.empty(b.shape[0],dtype=np.double)
            comm.Scatterv([b.T,MPI.DOUBLE],[b_loc,MPI.DOUBLE],root=0)
            c_loc=solve(a,b_loc,maxiter=maxiter,tol=tol)[0]
            c=np.zeros_like(b)
            comm.Gatherv([c_loc,MPI.DOUBLE],[c,MPI.DOUBLE],root=0)
        except Exception as e:
            if comm.rank==0:
                print('Parallel run failed: '+str(e))
            prun=False
    if not prun:
        c=np.array([solve(a,b[:,j],maxiter=maxiter,tol=tol)[0] for j in np.arange(b.shape[1])])
    if transp==prun: c=c.T
    return c
    
def matnrnd(M=None,U=1,V=1,n=1):
    """
    random sample from a matrix Normal distribution X ~ N_{I*J}(M,U,V)
    Z~N(0,I,I), U=L_U*L_U', V=R_V'*R_V, X = M + L_U * Z * R_V
    """
    I=U.shape[0]
    J=V.shape[0]
    K=M.shape[2] if M is not None and np.ndim(M)==3 else 1
    if M is not None:
        assert np.allclose([I,J],M.shape[:2]), 'Sizes not matched!'
    
    if not np.allclose(U,np.tril(U)):
        U=spla.cholesky(U,lower=True)
    if not np.allclose(V,np.triu(V)):
        V=spla.cholesky(V)
    X=np.random.randn(I,J,n)
    X=np.tensordot(np.tensordot(U,X,1),V,axes=(1,0)).swapaxes(1,2)
    if n==1: X=np.squeeze(X,2)
    if M is not None:
        if K==1:
            X+=M
        else:
            X+=M[:,:,np.resize(np.arange(K),n)]
    return X

def sparse_cholesky(A,**kwargs):
    """
    Cholesky decomposition for sparse matrix: the input matrix A must be a sparse symmetric positive semi-definite
    input: sparse symmetric positive-definite matrix A
    output: lower triangular factor L and pivot matrix P such that PLL^TP^T=A
    """
    n=A.shape[0]
    lu=spsla.splu(A,**kwargs)
    if ( lu.perm_r == lu.perm_c ).all() and ( lu.U.diagonal() >= 0 ).all(): # check the matrix A is positive semi-definite.
        L=lu.L.dot( sps.diags(lu.U.diagonal()**0.5) )
        P=sps.csc_matrix((np.ones(n),(np.arange(n),lu.perm_r)),shape=(n,)*2)
        return L,P
    else:
        raise Exception('The matrix is not positive semi-definite!')