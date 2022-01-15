"""
Main function to run Rossler inverse problem to generate posterior samples
Shiwei Lan @ ASU, 2021
"""

# modules
import os,argparse,pickle
import numpy as np
import dolfin as df

# the inverse problem
from Rossler import Rossler

# MCMC
import sys
sys.path.append( "../" )
from sampler.geoinfMC import geoinfMC

np.set_printoptions(precision=3, suppress=True)
seed=2021
np.random.seed(seed)

def main(seed=2021):
    parser = argparse.ArgumentParser()
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=2500)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.0001,.0001,.004,None,None]) # [.001,.005,.005] simple likelihood model
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('pCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()

    ## define Rossler inverse problem ##
    num_traj = 1
    t_init = 1000
    t_final = 1100
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = 'aug' # True; 'aug'; False
    var_out = True # True; 'cov'; False
    rsl = Rossler(num_traj=num_traj, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=False) # set STlik=False for simple likelihood; STlik has to be used with avg_traj
    
    # initialization
    # unknown=rsl.prior.sample(add_mean=True)
    unknown=np.log([.18, .18, 5.5])
    # MAP_file=os.path.join(os.getcwd(),'properties/MAP.pckl')
    # if os.path.isfile(MAP_file):
    #     f=open(MAP_file,'rb')
    #     unknown = pickle.load(f)
    #     f.close()
    # else:
    #     unknown=rsl.get_MAP(SAVE=True)
    
    # run MCMC to generate samples
    print("Preparing %s sampler with step size %g for %d step(s)..."
          % (args.algs[args.algNO],args.step_sizes[args.algNO],args.step_nums[args.algNO]))
    
    inf_GMC=geoinfMC(unknown,rsl,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO])
    # # force to update (b, c) only
    inf_GMC.q=unknown[1:]; inf_GMC.dim=2
    geom_ord=[0]
    if any(s in args.algs[args.algNO] for s in ['MALA','HMC']): geom_ord.append(1)
    def geom_(parameter):
        out = rsl.get_geom(np.insert(parameter,0,np.log(rsl.misfit.true_params['a'])),geom_ord=geom_ord)
        if 1 in geom_ord: out = (out[0], out[1][1:])
        return out
    inf_GMC.geom=geom_
    inf_GMC.ll, inf_GMC.g = inf_GMC.geom(inf_GMC.q)[:2]
    rsl.prior.mean=rsl.prior.mean[1:]; rsl.prior.std=rsl.prior.std[1:]; rsl.prior.d=2
    
    mc_fun=inf_GMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append ODE information including the count of solving
    filename_=os.path.join(inf_GMC.savepath,inf_GMC.filename+'.pckl')
    filename=os.path.join(inf_GMC.savepath,'Rossler_'+{True:'avg',False:'full','aug':'avgaug'}[rsl.misfit.avg_traj]+'_'+inf_GMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    soln_count=rsl.ode.soln_count
    pickle.dump([num_traj,obs_times,avg_traj,soln_count,args],f)
    f.close()
#     # verify with load
#     f=open(filename,'rb')
#     mc_samp=pickle.load(f)
#     pde_info=pickle.load(f)
#     f.close
#     print(pde_cnt)

if __name__ == '__main__':
    main()
    # set random seed
    # seeds = [2021+i*10 for i in range(1,10)]
    # n_seed = len(seeds)
    # for i in range(n_seed):
    #     print("Running for seed %d ...\n"% (seeds[i]))
    #     np.random.seed(seeds[i])
    #     main(seed=seeds[i])
