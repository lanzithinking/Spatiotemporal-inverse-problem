"""
Main function to run ensemble Kalman (EnK) algorithms for Lorenz96 inverse problem
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
    parser.add_argument('mdlNO', nargs='?', type=int, default=0)
    parser.add_argument('ensemble_size', nargs='?', type=int, default=100)
    parser.add_argument('max_iter', nargs='?', type=int, default=25)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1])
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    parser.add_argument('mdls', nargs='?', type=str, default=('simple','STlik'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(seed)
    
    ## define Lorenz96 inverse problem ##
    num_traj = 1 # only consider single trajectory!
    t_init = 100
    t_final = 110
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    K, L = 36, 10
    # try:
    #     f=open(os.path.join(os.getcwd(),'steady_state.pckl'),'rb')
    #     ode_init=pickle.load(f)
    #     print('Initialize Lorenz96 with steady state.')
    #     f.close()
    # except Exception:
    #     ode_init = -1 + 2*np.random.RandomState(2021).random((num_traj, K*(1+L)))
    #     ode_init[:,:K] *= 10
    avg_traj = {'simple':'aug','STlik':False}[args.mdls[args.mdlNO]] # True; 'aug'; False
    var_out = True # True; 'cov'; False
    STlik = (args.mdls[args.mdlNO]=='STlik')
    lrz96 = Lorenz96(num_traj=num_traj, obs_times=obs_times, K=K, L=L, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik)
    
    # initialization
    warm_start = False
    u0=lrz96.prior.sample(n=args.ensemble_size)
    if args.ensemble_size>200:
        n_jobs = np.min([10, multiprocessing.cpu_count()])
    if STlik:
        if args.ensemble_size<=200:
            G=lambda u:np.stack([lrz96.fwd(parameter=u_j, warm_start=warm_start).flatten() for u_j in u])
        else:
            G=lambda u:np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j:lrz96.fwd(parameter=u_j, warm_start=warm_start).flatten())(u_j) for u_j in u))
        y=lrz96.misfit.obs[0].flatten()
        nz_cov=lrz96.misfit.stgp.tomat()
    else:
        if args.ensemble_size<=200:
            G=lambda u:np.stack([lrz96.fwd(parameter=u_j, warm_start=warm_start).squeeze() for u_j in u])
        else:
            G=lambda u:np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j:lrz96.fwd(parameter=u_j, warm_start=warm_start).squeeze())(u_j) for u_j in u))
        y=lrz96.misfit.obs[0] #(1,5K)
        nz_cov=np.diag(lrz96.misfit.nzvar[0]) if np.ndim(lrz96.misfit.nzvar)==2 else lrz96.misfit.nzvar[0]
    data={'obs':y,'size':y.size,'cov':nz_cov}
    prior={'mean':lrz96.prior.mean,'cov':np.diag(lrz96.prior.var),'sample':lrz96.prior.sample}
    
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
    filename=os.path.join(savepath,'Lorenz96_'+{True:'avg',False:'full','aug':'avgaug'}[lrz96.misfit.avg_traj]+'_'+filename+'.pckl') # change filename
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