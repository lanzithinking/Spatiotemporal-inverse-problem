"""
Main function to run ensemble Kalman (EnK) algorithms for Lorenz96 inverse problem with different spin-up times and average time lengths.
Shiwei Lan @ ASU, 2021
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from Lorenz96 import Lorenz96

from joblib import Parallel, delayed
import multiprocessing

# EnK
import sys
sys.path.append( "../" )
from optimizer.EnK import *

np.set_printoptions(precision=3, suppress=True)
# seed=2021
# np.random.seed(seed)

def main(seed=2021, t0=100, t_res=100):
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('mdlNO', nargs='?', type=int, default=0)
    parser.add_argument('ensemble_size', nargs='?', type=int, default=500)
    parser.add_argument('max_iter', nargs='?', type=int, default=25)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1])
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    parser.add_argument('mdls', nargs='?', type=str, default=('simple','STlik'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(seed)
    
    ## define Lorenz96 inverse problem ##
    num_traj = 1 # only consider single trajectory!
    t_init = t0
    time_res = t_res
    dt = 0.1
    t_final = t_init + time_res*dt
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
    lrz96 = Lorenz96(num_traj=num_traj, obs_times=obs_times, K=K, L=L, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik,
                     use_saved_obs=True, save_obs=False) # set use_saved_obs=True when varying observation window size t_res
    
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
        y=lrz96.misfit.obs[0]
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
    enk_args=(args.max_iter,False)
    return_list=enk_fun(*enk_args)
    
    # append extra information
    return_list=return_list[:2]+return_list[3:]
    return_list+=(obs_times,avg_traj,STlik,var_out,y,args)
    # save
    savepath=savepath=os.path.join(os.getcwd(),args.mdls[args.mdlNO]+'_T')
    if not os.path.exists(savepath):
        print('Save path does not exist; created one.')
        os.makedirs(savepath)
    filename='Lorenz96_'+{True:'avg',False:'full','aug':'avgaug'}[lrz96.misfit.avg_traj]+'_'+ \
             enk.alg+'_ensbl'+str(enk.J)+'_dim'+str(enk.D)+'_Tinit'+str(t_init)+'_T'+str(time_res)+'_seed'+str(seed)
             # args.algs[args.algNO]+'_ensbl'+str(args.ensemble_size)+'_dim'+str(u0.shape[1])+'_Tinit'+str(t_init)+'_T'+str(time_res)+'_seed'+str(seed)          
    np.savez_compressed(os.path.join(savepath,filename), *return_list)

if __name__ == '__main__':
    # main()
    # run for multiple settings
    n_seed = 10; n_times=10
    for j in range(n_times):
        i=0; n_success=0
        while n_success < n_seed:
            seed_i=2021+i*10
            i+=1
            try:
                sep = "\n"+"*-"*40+"*\n"
                print(sep, "Running for time setting %d with seed %d ..."% (j, seed_i), sep)
                # main(seed=seed_i, t0=10*(j+1)) # vary spin-up t0 save and results in a folder ending with '_Tinit' (modify savepath)
                main(seed=seed_i, t_res=10*(j+1)) # vary window size T and save results in a folder ending with '_T' (modify savepath)
                n_success+=1
            except Exception as e:
                print(e)
                pass