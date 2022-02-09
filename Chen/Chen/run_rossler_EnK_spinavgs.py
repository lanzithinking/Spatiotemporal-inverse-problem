"""
Main function to run ensemble Kalman (EnK) algorithms for Rossler inverse problem with different spin-up times and average time lengths.
Shiwei Lan @ ASU, 2021
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from Rossler import Rossler

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
    parser.add_argument('max_iter', nargs='?', type=int, default=50)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1.,.1])
    parser.add_argument('algs', nargs='?', type=str, default=('EKI','EKS'))
    parser.add_argument('mdls', nargs='?', type=str, default=('simple','STlik'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(seed)

    ## define Rossler inverse problem ##
    num_traj = 1 # only consider single trajectory!
    prior_params = {'mean':[-1.5, -1.5, 2], 'std':[0.15, 0.15, 0.2]}
    t_init = t0
    time_res = t_res
    dt = 1
    t_final = t_init + time_res*dt
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = {'simple':'aug','STlik':False}[args.mdls[args.mdlNO]] # True; 'aug'; False
    var_out = True # True; 'cov'; False
    STlik = (args.mdls[args.mdlNO]=='STlik')
    rsl = Rossler(num_traj=num_traj, prior_params=prior_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik,
                  use_saved_obs=False, save_obs=False) # set STlik=False for simple likelihood; STlik has to be used with avg_traj
    
    # initialization
    u0=rsl.prior.sample(n=args.ensemble_size)
    # G=lambda u:np.stack([rsl.misfit.observe(sol=rsl.ode.solve(params=np.exp(u_j), t=rsl.obs_times)).squeeze() for u_j in u])
    if args.ensemble_size>200:
        n_jobs = np.min([10, multiprocessing.cpu_count()])
        # G=lambda u:np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j:rsl.misfit.observe(sol=rsl.ode.solve(params=np.exp(u_j), t=rsl.obs_times)).squeeze())(u_j) for u_j in u))
    # y=rsl.misfit.obs[0]
    if STlik:
        if args.ensemble_size<=200:
            G=lambda u:np.stack([rsl.misfit.observe(sol=rsl.ode.solve(params=np.exp(u_j), t=rsl.obs_times)).flatten() for u_j in u])
        else:
            G=lambda u:np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j:rsl.misfit.observe(sol=rsl.ode.solve(params=np.exp(u_j), t=rsl.obs_times)).flatten())(u_j) for u_j in u))
        y=rsl.misfit.obs[0].flatten()
        nz_cov=rsl.misfit.stgp.tomat()
    else:
        if args.ensemble_size<=200:
            G=lambda u:np.stack([rsl.misfit.observe(sol=rsl.ode.solve(params=np.exp(u_j), t=rsl.obs_times)).squeeze() for u_j in u])
        else:
            G=lambda u:np.stack(Parallel(n_jobs=n_jobs)(delayed(lambda u_j:rsl.misfit.observe(sol=rsl.ode.solve(params=np.exp(u_j), t=rsl.obs_times)).squeeze())(u_j) for u_j in u))
        y=rsl.misfit.obs[0]
        nz_cov=np.diag(rsl.misfit.nzvar[0]) if np.ndim(rsl.misfit.nzvar)==2 else rsl.misfit.nzvar[0]
    data={'obs':y,'size':y.size,'cov':nz_cov}
    prior={'mean':rsl.prior.mean,'cov':np.diag(rsl.prior.std)**2,'sample':rsl.prior.sample}
    
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
    savepath=savepath=os.path.join(os.getcwd(),args.mdls[args.mdlNO]+'_Tinit')
    if not os.path.exists(savepath):
        print('Save path does not exist; created one.')
        os.makedirs(savepath)
    filename='Rossler_'+{True:'avg',False:'full','aug':'avgaug'}[rsl.misfit.avg_traj]+'_'+ \
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
                main(seed=seed_i, t0=10*(j+1))
                # main(seed=seed_i, t_res=10*(j+1))
                n_success+=1
            except Exception as e:
                print(e)
                pass