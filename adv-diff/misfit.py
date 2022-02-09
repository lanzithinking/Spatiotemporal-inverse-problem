'''
Misfit of the Advection-Diffusion problem written in FEniCS-2019.1.0 and hIPPYlib-3.0
https://hippylib.github.io/tutorials_v3.0.0/4_AdvectionDiffusionBayesian/
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)
Shiwei Lan @ ASU, Sept. 2020
--------------------------------------------------------------------------
Created on Sep 23, 2020
-------------------------------
Modified August 15, 2021 @ ASU
-------------------------------
https://github.com/lanzithinking/Spatiotemporal-inverse-problem
'''
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt

import sys
import os, pickle
# sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
from pde import TimeDependentAD

sys.path.append( "../" )
from util.common_colorbar import common_colorbar
from util.stgp.GP import GP
from util.stgp.STGP import STGP
from util.stgp.STGP_mg import STGP_mg

class SpaceTimePointwiseStateObservation(Misfit):
    """
    Misfit (negative loglikelihood) of Advection-Diffusion inverse problem
    """
    def __init__(self, Vh, observation_times=None, targets=None, rel_noise = 0.01, d = None, **kwargs):
        """
        Initialize the misfit
        """
        # function space
        self.Vh = Vh
        self.mpi_comm = self.Vh.mesh().mpi_comm()
        self.rank = dl.MPI.rank(self.mpi_comm)
        # observation times
        if observation_times is None:
            t_init         = 0.
            t_final        = 4.
            t_1            = 1.
            dt             = .1
            observation_dt = .2
            self.observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)
        else:
            self.observation_times = observation_times
        # observation locations
        self.targets = np.loadtxt('targets.txt') if targets is None else targets
        self.rel_noise = rel_noise
        # obtain observations
        if d is None:
            d=self.get_observations(**kwargs)
            if self.rank == 0:
                sep = "\n"+"#"*80+"\n"
                print( sep, "Generate synthetic observations at {0} locations for {1} time points".format(self.targets.shape[0], len(self.observation_times)), sep )
        # reset observation container for reference
        self.prep_container()
        self.d.axpy(1., d)
        
        self.STlik = kwargs.pop('STlik',True)
        
        
        if self.STlik:
            self.stgp = kwargs.get('stgp')
            if self.stgp is None:
                # define STGP kernel for the likelihood (misfit)
                # self.stgp=STGP(spat=self.targets, temp=self.observation_times, opt=kwargs.pop('ker_opt',0), jit=1e-2)
                C_x=GP(self.targets, l=.5, sigma2=np.sqrt(self.noise_variance), store_eig=True, jit=1e-2)
                C_t=GP(self.observation_times, store_eig=True, l=.2, sigma2=np.sqrt(self.noise_variance))#, ker_opt='matern',nu=.5)
                # C_x=GP(self.targets, l=.4, jit=1e-3, sigma2=.1, store_eig=True)
                # C_t=GP(self.observation_times, store_eig=True, l=.2, sigma2=.1, ker_opt='matern',nu=.5)
                self.stgp=STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',0), spdapx=False)
                # C_x=GP(self.targets, l=.5, sigma2=.1, store_eig=True)
                # C_t=GP(self.observation_times, store_eig=True, l=.2, sigma2=1.)#, ker_opt='matern',nu=.5)
                # self.stgp=STGP_mg(STGP(spat=C_x, temp=C_t, opt=kwargs.pop('ker_opt',2), spdapx=False), K=1, nz_var=self.noise_variance, store_eig=True)
        
    def prep_container(self, Vh=None):
        """
        Prepare storage of the observations
        """
        if Vh is None:
            Vh = self.Vh
        # storage for observations
        self.B = assemblePointwiseObservation(Vh, self.targets)
        self.d = TimeDependentVector(self.observation_times)
        self.d.initialize(self.B, 0)
        ## TEMP Vars
        self.u_snapshot = dl.Vector()
        self.Bu_snapshot = dl.Vector()
        self.d_snapshot  = dl.Vector()
        self.B.init_vector(self.u_snapshot, 1)
        self.B.init_vector(self.Bu_snapshot, 0)
        self.B.init_vector(self.d_snapshot, 0)
    
    def get_observations(self, **kwargs):
        """
        Get the observations at given locations and time points
        """
        fld=kwargs.pop('obs_file_loc',os.getcwd())
        try:
            f=open(os.path.join(fld,'AdvDiff_obs.pckl'),'rb')
            obs,self.noise_variance=pickle.load(f)
            f.close()
            self.prep_container()
            for i,t in enumerate(self.observation_times):
                self.d_snapshot.set_local(obs[i])
                self.d.store(self.d_snapshot,t)
            print('Observation file has been read!')
        except Exception as e:
            print(e)
            pde=kwargs.pop('pde',None)
            nref=kwargs.pop('nref',0)
            init=kwargs.pop('init',None)
            # pde for observations
            if pde is None:
                mesh = self.Vh.mesh()
                for i in range(nref): mesh = dl.refine(mesh) # refine mesh to obtain observations
                pde = TimeDependentAD(mesh)
            elif nref>0:
                mesh = pde.mesh
                for i in range(nref): mesh = dl.refine(mesh) # refine mesh to obtain observations
                pde = TimeDependentAD(mesh)
            # initial condition
            if init is None:
                true_init = dl.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=pde.Vh[STATE].ufl_element())
                init = dl.interpolate(true_init, pde.Vh[STATE]).vector()
            # prepare container for observations
            self.prep_container(pde.Vh[STATE])
            
            utrue = pde.generate_vector(STATE)
            x = [utrue, init, None]
            pde.solveFwd(x[STATE], x)
            self.observe(x, self.d)
            MAX = self.d.norm("linf", "linf")
            noise_std_dev = self.rel_noise * MAX
            parRandom.normal_perturb(noise_std_dev,self.d)
            self.noise_variance = noise_std_dev*noise_std_dev
            
            # save
            if kwargs.pop('save_obs',True):
                obs = []
                for t in self.observation_times:
                    self.d.retrieve(self.d_snapshot, t)
                    obs.append(self.d_snapshot.get_local())
                obs = np.stack(obs)
                f=open(os.path.join(fld,'AdvDiff_obs.pckl'),'wb')
                pickle.dump([obs,self.noise_variance],f)
                f.close()
        return self.d.copy()
    
    def observe(self, x, obs=None):
        """
        Observation operator
        """
        if obs is None:
            obs = []
        else:
            obs.zero()
        
        for t in self.observation_times:
            x[STATE].retrieve(self.u_snapshot, t)
            self.B.mult(self.u_snapshot, self.Bu_snapshot)
            obs.store(self.Bu_snapshot, t) if type(obs) is TimeDependentVector else obs.append(self.Bu_snapshot.get_local())
        
        if type(obs) is list: return np.stack(obs)
            
    def cost(self, x, option='nll'):
        """
        Compute misfit
        option: return negative loglike ('nll') or (postive) quadratic form (quad) where loglike = halfdelt+quad
        """
        if self.STlik:
            du = []
            for t in self.observation_times:
                x[STATE].retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.d.retrieve(self.d_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.d_snapshot)
                du.append(self.Bu_snapshot.get_local())
            du = np.stack(du).T # (I,J)
            # if option=='nll':
            #     res = -self.stgp.matn0pdf(du)[0]#,nu=self.noise_variance)[0]
            # elif option=='quad': #1/2(y-G(u))'C^(-1)(y-G(u))
            #     logpdf,half_ldet = self.stgp.matn0pdf(du)
            #     res = -(logpdf - half_ldet)
            logpdf,half_ldet = self.stgp.matn0pdf(du)
            res = {'nll':-logpdf, 'quad':-(logpdf - half_ldet), 'both':[-logpdf,-(logpdf - half_ldet)]}[option]
        else:
            c = 0
            for t in self.observation_times:
                x[STATE].retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot)
                self.d.retrieve(self.d_snapshot, t)
                self.Bu_snapshot.axpy(-1., self.d_snapshot)
                c += self.Bu_snapshot.inner(self.Bu_snapshot)
            res = c/(2.*self.noise_variance)
            
        return  res
    
    def grad(self, i, x, out):
        """
        Compute the gradient of misfit
        """
        out.zero()
        if i == STATE:
            if self.STlik:
                du = []
                for t in self.observation_times:
                    x[STATE].retrieve(self.u_snapshot, t)
                    self.B.mult(self.u_snapshot, self.Bu_snapshot)
                    self.d.retrieve(self.d_snapshot, t)
                    self.Bu_snapshot.axpy(-1., self.d_snapshot)
                    # self.Bu_snapshot *= 1./self.noise_variance
                    du.append(self.Bu_snapshot.get_local())
                du = np.stack(du).T
                g = self.stgp.solve(du).reshape(du.shape,order='F')
                for j,t in enumerate(self.observation_times):
                    self.Bu_snapshot.set_local(g[:,j])
                    self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                    out.store(self.u_snapshot, t)
            else:
                for t in self.observation_times:
                    x[STATE].retrieve(self.u_snapshot, t)
                    self.B.mult(self.u_snapshot, self.Bu_snapshot)
                    self.d.retrieve(self.d_snapshot, t)
                    self.Bu_snapshot.axpy(-1., self.d_snapshot)
                    self.Bu_snapshot *= 1./self.noise_variance
                    self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                    out.store(self.u_snapshot, t)
        else:
            pass
    
    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass
    
    def apply_ij(self, i,j, direction, out):
        out.zero()
        if i == STATE and j == STATE:
            if self.misfit.STlik:
                du = []
                for t in self.observation_times:
                    direction.retrieve(self.u_snapshot, t)
                    self.B.mult(self.u_snapshot, self.Bu_snapshot)
                    # self.Bu_snapshot *= 1./self.noise_variance
                    du.append(self.Bu_snapshot.get_local())
                du = np.stack(du).T
                g = self.misfit.stgp.solve(du).reshape(du.shape,order='F')
                for j,t in enumerate(self.observation_times):
                    self.Bu_snapshot.set_local(g[:,j])
                    self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                    out.store(self.u_snapshot, t)
            else:
                for t in self.observation_times:
                    direction.retrieve(self.u_snapshot, t)
                    self.B.mult(self.u_snapshot, self.Bu_snapshot)
                    self.Bu_snapshot *= 1./self.noise_variance
                    self.B.transpmult(self.Bu_snapshot, self.u_snapshot) 
                    out.store(self.u_snapshot, t)
        else:
            pass    
    
    def applyWuu(self, du, out):
        out.zero()
        self.apply_ij(STATE, STATE, du, out)
    
    def applyWum(self, dm, out):
        out.zero()
    
    def applyWmu(self, du, out):
        out.zero()
    
    def applyWmm(self, dm, out):
        out.zero()
    
    def plot_data(self, times, figsz=(12,5)):
        """
        Plot the observations with its values u(x, t) at fixed locations for given time points
        """
        n=len(times)
        nrow=np.floor(np.sqrt(n)).astype('int')
        ncol=np.ceil(np.sqrt(n)).astype('int')
        fig,axes=plt.subplots(nrows=nrow,ncols=ncol,sharex=True,sharey=True,figsize=figsz)
        sub_figs = [None]*len(axes.flat)
        for i in range(n):
            plt.axes(axes.flat[i])
            dl.plot(self.Vh.mesh())
            sub_figs[i]=plt.scatter(self.targets[:,0],self.targets[:,1], s=12, c=self.d.data[np.where(np.isclose(self.d.times,times[i]))[0][0]], zorder=2)
#             plt.xlim(0,1); plt.ylim(0,1)
#             plt.gca().set_aspect('equal', 'box')
            plt.title('Time: {:.1f} s'.format(times[i],))
        fig=common_colorbar(fig,axes,sub_figs)
        return fig
    
if __name__ == '__main__':
    np.random.seed(2020)
    from prior import *
#     # define pde
    meshsz = (61,61)
    eldeg = 1
    pde = TimeDependentAD(mesh=meshsz, eldeg=eldeg)
    Vh = pde.Vh[STATE]
    # obtain function space
#     mesh = dl.Mesh('ad_10k.xml')
#     Vh = dl.FunctionSpace(mesh, "Lagrange", 2)
    # set observation times
    t_init         = 0.
    t_final        = 4.
    t_1            = 1.
    dt             = .1
    observation_dt = .2
    observation_times = np.arange(t_1, t_final+.5*dt, observation_dt)
    # set observation locations
    targets = np.loadtxt('targets.txt')
    # define misfit
    rel_noise = .5
    nref = 1
    misfit = SpaceTimePointwiseStateObservation(Vh, observation_times, targets, rel_noise=rel_noise, nref=nref)
#     # optional: refine mesh to obtain (new) observations
#     rf_mesh = dl.refine(pde.mesh)
#     rf_pde = TimeDependentAD(mesh=rf_mesh)
#     rf_obs = SpaceTimePointwiseStateObservation(rf_pde.Vh[STATE], observation_times, targets, pde=rf_pde).d.copy()
#     misfit.d.zero()
#     misfit.d.axpy(1.,rf_obs)
    # plot observations
    plt.rcParams['image.cmap'] = 'jet'
    plt_times=[1.,2.,3.,4.]
    fig = misfit.plot_data(plt_times, (9,8))
    plt.subplots_adjust(wspace=-0.1, hspace=0.2)
    plt.savefig(os.path.join(os.getcwd(),'properties/obs.png'),bbox_inches='tight')
    
    # test gradient
    prior = BiLaplacian(Vh=pde.Vh[PARAMETER], gamma=1., delta=8.)
    u = prior.sample()
    x = [pde.generate_vector(STATE), u, None]
    pde.solveFwd(x[STATE], x)
    c = misfit.cost(x)
    g = pde.generate_vector(STATE)
    misfit.grad(STATE, x, g)
    v = prior.sample()
    dx = [pde.generate_vector(STATE), v, None]
    pde.solveFwd(dx[STATE], dx)
    h = 1e-8
    x[STATE].axpy(h,dx[STATE])
    c1 = misfit.cost(x)
    gdx_fd = (c1-c)/h
    gdx = g.inner(dx[STATE])
    rdiff_gdx = abs(gdx_fd-gdx)/dx[STATE].norm("linf", 'l2')
    print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gdx)