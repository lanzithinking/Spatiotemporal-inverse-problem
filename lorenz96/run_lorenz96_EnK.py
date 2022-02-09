"""
Main function to run ensemble Kalman (EnK) algorithms for Rossler inverse problem
Shiwei Lan @ ASU, 2021
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from Lorenz96 import *

from joblib import Parallel, delayed
import multiprocessing

# EnK
import sys
sys.path.append( "../" )
from optimizer.EnK import *

np.set_printoptions(precision=3, suppress=True)
# seed=2021
# np.random.seed(seed)

def main(seed=2021):
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('mdlNO', nargs='?', type=int, default=1)
    parser.add_argument('ensemble_size', nargs='?', type=int, default=100)
    parser.add_argument('max_iter', nargs='?', type=int, default=50)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1])
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    parser.add_argument('mdls', nargs='?', type=str, default=('simple','STlik'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(seed)  
    ## define lorenz96 inverse problem ##
    num_traj = 1
    t_init = 0
    t_final = 10
    time_res = 200
    obs_times = np.linspace(t_init, t_final, time_res)
    L, K = 5, 8
    n = (L+1) * K
    avg_traj = {'simple':True,'STlik':False}[args.mdls[args.mdlNO]] # True; False
    var_out = True #'cov' ; False   
    STlik = (args.mdls[args.mdlNO]=='STlik')
    
    lorenz96 = Lorenz96(obs_times=obs_times, L=L, K=K, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik)
    
        
    # initialization
    u0=lorenz96.prior.sample(n=args.ensemble_size)
    # G=lambda u:np.stack([lorenz96.misfit.observe(sol=lorenz96.ode.solve(params=np.exp(u_j), t=lorenz96.obs_times)).squeeze() for u_j in u])
    if args.ensemble_size>200:
        n_jobs = np.min([10, multiprocessing.cpu_count()])
        # G=lambda u:np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j:lorenz96.misfit.observe(sol=lorenz96.ode.solve(params=np.exp(u_j), t=lorenz96.obs_times)).squeeze())(u_j) for u_j in u))
    # y=lorenz96.misfit.obs[0]
    if STlik:
        if args.ensemble_size<=200:
            G=lambda u:np.stack([lorenz96.misfit.observe(sol=lorenz96.ode.solve(params=(u_j), t=lorenz96.obs_times)).flatten() for u_j in u])
        else:
            G=lambda u:np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j:lorenz96.misfit.observe(sol=lorenz96.ode.solve(params=(u_j), t=lorenz96.obs_times)).flatten())(u_j) for u_j in u))
        y=lorenz96.misfit.obs.flatten()
        nz_cov=lorenz96.misfit.stgp.tomat()
    else:
        if args.ensemble_size<=200:
            G=lambda u:np.stack([lorenz96.misfit.observe(sol=lorenz96.ode.solve(params=(u_j), t=lorenz96.obs_times)).squeeze() for u_j in u])
        else:
            G=lambda u:np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j:lorenz96.misfit.observe(sol=lorenz96.ode.solve(params=(u_j), t=lorenz96.obs_times)).squeeze())(u_j) for u_j in u))
        y=lorenz96.misfit.obs #(K*5,)
        nz_cov=np.diag(lorenz96.misfit.nzvar[0]) if np.ndim(lorenz96.misfit.nzvar)==2 else lorenz96.misfit.nzvar[0]
    data={'obs':y,'size':y.size,'cov':nz_cov}
    prior={'mean':lorenz96.prior.mean,'cov':np.diag(lorenz96.prior.std)**2,'sample':lorenz96.prior.sample}
    
    # EnK parameters
    nz_lvl=1.0
    err_thld=1e-2
    
    # run EnK to generate ensembles
    print("Preparing %s with step size %g for %s model..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.mdls[args.mdlNO]))
    enk=EnK(u0,G,data,prior,stp_sz=args.step_sizes[args.algNO],nz_lvl=nz_lvl,err_thld=err_thld,alg=args.algs[args.algNO],adpt=True)
    enk_fun=enk.run
    enk_args=(args.max_iter,True)
    savepath,filename=enk_fun(*enk_args)
    
    # append extra information including the count of solving
    filename_=os.path.join(savepath,filename+'.pckl')
    filename=os.path.join(savepath,'Lorenz96_'+{True:'avg',False:'full','aug':'avgaug'}[rsl.misfit.avg_traj]+'_'+filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    pickle.dump([obs_times,avg_traj,STlik,var_out,y,args],f)
    f.close()

if __name__ == '__main__':
    # main()
    # set random seed
    # seeds = [2021+i*10**(1+args.algNO) for i in range(10)]
    # n_seed = len(seeds)
    # for i in range(n_seed):
        # print("Running for seed %d ...\n"% (seeds[i]))
        # np.random.seed(seeds[i])
        # main(seed=seeds[i])
    n_seed = 10; i=0; n_success=0
    while n_success < n_seed:
        i+=1
        seed_i=2021+i*10
        try:
            print("Running for seed %d ...\n"% (seed_i))
            main(seed=seed_i)
            n_success+=1
        except Exception as e:
            print(e)
            pass