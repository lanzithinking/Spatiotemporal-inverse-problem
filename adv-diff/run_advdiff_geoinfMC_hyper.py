"""
Main function to run Advection-Diffusion inverse problem to generate posterior samples
Shiwei Lan @ ASU, 2020
----------------------
Created by Shuyi Li, Sept. 2021
"""

# modules
import os,argparse,pickle
import numpy as np
import dolfin as df
import time,timeit
import scipy.stats as spst
from scipy.optimize import minimize

# the inverse problem
from advdiff import advdiff
STATE = 0; PARAMETER = 1

# hyper-parameters
from opt4ini import opt4ini
from logpdf_hyperpars import *

# MCMC
import sys
sys.path.append( "../" )
from sampler.geoinfMC_dolfin import geoinfMC
from sampler.slice import slice as slice_sampler

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)
seed=2020
np.random.seed(seed)

def main(seed=2020):
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=5000)
    parser.add_argument('thin', nargs='?', type=int, default=5)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.01,.01,.01,None,None]) # [.001,.005,.005] simple likelihood model
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()
    upthypr = 1 # 0: no update; 1: sample; 2; optimize; 3: optimize jointly
    
    ## define Advection-Diffusion inverse problem ##
    # mesh = df.Mesh('ad_10k.xml')
    meshsz = (61,61)
    eldeg = 1
    gamma = 2.; delta = 10.
    rel_noise = .5
    nref = 1
    adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed, STlik=True)
    adif.prior.V=adif.prior.Vh
    
    # specify (hyper)-priorsx
    # (a,b) in inv-gamma priors for sigma2_t
    # (m,V) in (log) normal priors for eta_*, (eta=log-rho), * = x, t
    a = 2.
    b = 1e-1
    m = [0,0]
    V = np.asarray([1.,10.])
    if upthypr:
        opts_unc={'gtol': 1e-6, 'disp': False, 'maxiter': 100}
    
    # initialization
    sigma2=spst.invgamma.rvs(a, scale=b);# 1/spst.gamma.rvs(a, scale=1/b);for sigma2_t
    eta=spst.norm.rvs(m,np.sqrt(V));
    adif.misfit.stgp.update(C_x=adif.misfit.stgp.C_x.update(sigma2 = 1., l = np.exp(eta[0])),
                            C_t=adif.misfit.stgp.C_t.update(sigma2 = sigma2, l = np.exp(eta[1])))
    dlta=adif.misfit.stgp.I*adif.misfit.stgp.J/2;
    alpha = a+dlta;
    # unknown=adif.prior.gen_vector()
    unknown=adif.prior.sample(whiten=False)
    # MAP_file=os.path.join(os.getcwd(),'properties/MAP.xdmf')
    # if os.path.isfile(MAP_file):
    #     unknown=df.Function(adif.prior.V, name='MAP')
    #     f=df.XDMFFile(adif.mpi_comm,MAP_file)
    #     f.read_checkpoint(unknown,'m',0)
    #     f.close()
    #     unknown = unknown.vector()
    # else:
    #     unknown=adif.get_MAP(SAVE=True)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    inf_GMC=geoinfMC(unknown,adif,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO])
    # original sampling 
    # mc_fun=inf_GMC.sample
    # mc_args=(args.num_samp,args.num_burnin)
    # mc_fun(*mc_args)
    
    ######  MCMC  ############################################################################
    # optimize initial values
    # sigma2,eta,inf_GMC,optimf = opt4ini(sigma2,eta,inf_GMC,a,b,m,V,jtopt=upthypr==3,Nmax=20);
    
    # allocate space to store results
    num_iters=args.num_samp*args.thin+args.num_burnin
    samp_sigma2=np.zeros((args.num_samp,1))
    samp_eta=np.zeros((args.num_samp,2))
    engy=np.zeros((num_iters,3))
    samp_fname='_samp_'+inf_GMC.alg_name+'_dim'+str(inf_GMC.dim)+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")
    samp_fpath=os.path.join(os.getcwd(),'result')
    if not os.path.exists(samp_fpath):
        os.makedirs(samp_fpath)
    #         inf_GMC.samp=df.File(os.path.join(samp_fpath,samp_fname+".xdmf"))
    inf_GMC.samp=df.HDF5File(inf_GMC.model.pde.mpi_comm,os.path.join(samp_fpath,samp_fname+".h5"),"w")
    inf_GMC.loglik=np.zeros(num_iters)
    inf_GMC.acpt=0.0 # final acceptance rate
    inf_GMC.times=np.zeros(num_iters) # record the history of time used for each sample
    
    """
    sample with given MCMC method
    """
    name_sampler = str(inf_GMC.alg_name)
    try:
        sampler = getattr(inf_GMC, name_sampler)
    except AttributeError:
        print(inf_GMC.alg_name, 'not found!')
    else:
        print('\nRunning '+inf_GMC.alg_name+' now...\n')
    
    # number of adaptations for step size
    if inf_GMC.adpt_h:
        inf_GMC.h_adpt['n_adpt']=args.num_burnin#kwargs.pop('adpt_steps',args.num_burnin)
        
    # online parameters
    accp=0.0 # online acceptance
    num_cons_bad=0 # number of consecutive bad proposals
    num_retry_bad=0
    
    beginning=timeit.default_timer()
    for s in range(num_iters):

        if s==args.num_burnin:
            # start the timer
            tic=timeit.default_timer()
            print('\nBurn-in completed; recording samples now...\n')
        
        # update sigma2
        dltb = inf_GMC.model.misfit.cost(inf_GMC.model.x, option='quad')*sigma2
        beta = b+dltb; nl_sigma2=0
        if upthypr==1:
            sigma2=spst.invgamma.rvs(a, scale=beta); #1/spst.gamma.rvs(alpha,scale=1/beta);
            # nl_sigma2=-(spst.gamma.logpdf(1/sigma2,alpha,scale=1/beta)-2*np.log(sigma2));
            nl_sigma2=-spst.invgamma.logpdf(sigma2,alpha,scale=beta)
        elif upthypr>=2:
            sigma2=beta/(alpha+1); # optimize
            # nl_sigma2=-(spst.gamma.logpdf(1/sigma2,alpha,scale=1/beta)-2*np.log(sigma2));
            nl_sigma2=-spst.invgamma.logpdf(sigma2,alpha,scale=beta)
        if upthypr:
            inf_GMC.model.misfit.stgp.update(C_t=inf_GMC.model.misfit.stgp.C_t.update(sigma2 = sigma2))
        
        # update eta
        logf=[]; nl_eta=np.zeros(2)
        # eta_x
        logf.append(lambda q: logpost_eta(q,inf_GMC,m[0],V[0],[0]))
        if upthypr==1:
            eta[0], l_eta = slice_sampler(eta[0],logf[0](eta[0]),logf[0]);
            nl_eta[0] = -l_eta
        elif upthypr==2:
            res=minimize(lambda q: -logf[0](q),eta[0],method='BFGS',options=opts_unc);
            eta[0], nl_eta[0] = res.x,res.fun
        # eta_t
        logf.append(lambda q: logpost_eta(q,inf_GMC,m[1],V[1],[1]))
        if upthypr==1:
            eta[1], l_eta = slice_sampler(eta[1],logf[1](eta[1]),logf[1]);
            nl_eta[1] = -l_eta
        elif upthypr==2:
            res=minimize(lambda q: -logf[1](q),eta[1],method='BFGS',options=opts_unc);
            eta[1], nl_eta[1] = res.x,res.fun
        # joint optimize
        if upthypr==3:
            logF=lambda q: logpost_eta(q,inf_GMC,m,V,[0,1]).sum()
            res=minimize(lambda q: -logF(q),eta,method='BFGS',options=opts_unc);
            eta, nl_eta = res.x,res.fun
            nl_eta = (nl_eta,np.nan)
        if upthypr:
            inf_GMC.model.misfit.stgp.update(C_x=inf_GMC.model.misfit.stgp.C_x.update(l = np.exp(eta[0])),
                                             C_t=inf_GMC.model.misfit.stgp.C_t.update(l = np.exp(eta[1])))
        
        # generate MCMC sample of unknown parameter with given sampler
        while True:
            try:
                acpt_idx,logr=sampler()
            except RuntimeError as e:
                print(e)
                # import pydevd; pydevd.settrace()
                # import traceback; traceback.print_exc()
                if num_retry_bad==0:
                    acpt_idx=False; logr=-np.inf
                    print('Bad proposal encountered! Passing... bias introduced.')
                    break # reject bad proposal: bias introduced
                else:
                    num_cons_bad+=1
                    if num_cons_bad<num_retry_bad:
                        print('Bad proposal encountered! Retrying...')
                        continue # retry until a valid proposal is made
                    else:
                        acpt_idx=False; logr=-np.inf # reject it and keep going
                        num_cons_bad=0
                        print(str(num_retry_bad)+' consecutive bad proposals encountered! Passing...')
                        break # reject it and keep going
            else:
                num_cons_bad=0
                break
        accp+=acpt_idx
        # update unknown parameter and forward output 
        # inf_GMC.model.x[PARAMETER] = inf_GMC.q # already updated internally in infGMC
        # inf_GMC.model.pde.solveFwd(inf_GMC.model.x[STATE], inf_GMC.model.x)
        
        # display acceptance at intervals
        if (s+1)%100==0:
            print('\nAcceptance at %d iterations: %0.2f' % (s+1,accp/100))
            accp=0.0
        
        # save results
        inf_GMC.loglik[s]=inf_GMC.ll # inf_GMC.geom(inf_GMC.q)[0]
        engy[s] = np.concatenate(((nl_sigma2,), nl_eta))
        if s>=args.num_burnin and (s-args.num_burnin)%args.thin==0:
            NO_sav=(s-args.num_burnin)//args.thin
            samp_sigma2[NO_sav] = sigma2
            samp_eta[NO_sav] = eta
            # inf_GMC.samp << vec2fun(inf_GMC.q,inf_GMC.model.prior.V,name='sample_{0}'.format(s-args.num_burnin))
            q_f=df.Function(inf_GMC.model.prior.V)
            # q_f.vector()[:]=inf_GMC.q
            q_f.vector().set_local(inf_GMC.q)
            q_f.vector().apply('insert')
            # q_f.vector().zero()
            # q_f.vector().axpy(1.,inf_GMC.q)
            inf_GMC.samp.write(q_f,'sample_{0}'.format(NO_sav))
            inf_GMC.acpt+=acpt_idx
        
        # record the time
        inf_GMC.times[s]=timeit.default_timer()-beginning
        
        # adapt step size h if needed
        if inf_GMC.adpt_h:
            if s<inf_GMC.h_adpt['n_adpt']:
                inf_GMC.h_adpt=inf_GMC._dual_avg(s+1,np.exp(min(0,logr)))
                inf_GMC.h=inf_GMC.h_adpt['h']
                print('New step size: %.2f; \t New averaged step size: %.6f\n' %(inf_GMC.h_adpt['h'],np.exp(inf_GMC.h_adpt['loghn'])))
            if s==inf_GMC.h_adpt['n_adpt']:
                inf_GMC.h_adpt['h']=np.exp(inf_GMC.h_adpt['loghn'])
                inf_GMC.h=inf_GMC.h_adpt['h']
                print('Adaptation completed; step size freezed at:  %.6f\n' % inf_GMC.h_adpt['h'])
    
    # stop timer
    inf_GMC.samp.close()
    toc=timeit.default_timer()
    inf_GMC.time=toc-tic
    inf_GMC.acpt/=args.num_samp
    print("\nAfter %g seconds, %d samples have been collected with the final acceptance rate %0.2f \n"
          % (inf_GMC.time,args.num_samp,inf_GMC.acpt))
    
    # save to file
    inf_GMC.save_samp()
    #### MCMC(done) ############################################################################
    
    # append PDE information including the count of solving
    filename_=os.path.join(inf_GMC.savepath,inf_GMC.filename+'.pckl')
    filename=os.path.join(inf_GMC.savepath,'AdvDiff_'+inf_GMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
#     soln_count=[adif.soln_count,adif.pde.soln_count]
    soln_count=adif.pde.soln_count
    pickle.dump([samp_sigma2,samp_eta,engy,meshsz,rel_noise,nref,soln_count,args],f)
    f.close()
#     # verify with load
#     f=open(filename,'rb')
#     mc_samp=pickle.load(f)
#     pde_info=pickle.load(f)
#     f.close
#     print(pde_cnt)

if __name__ == '__main__':
    main()
    # # set random seed
    # seeds = [2020+i*10 for i in range(1,10)]
    # n_seed = len(seeds)
    # for i in range(n_seed):
    #     print("Running for seed %d ...\n"% (seeds[i]))
    #     np.random.seed(seeds[i])
    #     main(seed=seeds[i])
