"""
To plot pairwise density
Shiwei Lan @ ASU, 2022
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Rossler import *

from joblib import Parallel, delayed
import multiprocessing

# define marginal density plot
def plot_pdf(x, **kwargs):
    nx = len(x)
    # z = np.zeros(nx)
    para0 = kwargs.pop('para0',None)
    f = kwargs.pop('f',None)
    # for i in range(nx):
    def parfor(i):
        para_ = para0.copy()
        para_[x.name] = x[i]
        # z[i] = f(list(para_.values()))
        return f(list(para_.values()))
    n_jobs = np.min([5, multiprocessing.cpu_count()])
    z = Parallel(n_jobs=n_jobs)(delayed(parfor)(i) for i in range(nx))
    z = np.array(z)
    
    plt.plot(x, z, **kwargs)

# define contour function
def contour(x, y, **kwargs):
    nx = len(x); ny = len(y)
    # z = np.zeros((nx, ny))
    para0 = kwargs.pop('para0',None)
    f = kwargs.pop('f',None)
    # for i in range(nx):
        # for j in range(ny):
    def parfor(i, j):
        para_ = para0.copy()
        para_[x.name] = x[i]; para_[y.name] = y[j]
            # z[i,j] = f(list(para_.values()))
        return f(list(para_.values()))
    n_jobs = np.min([10, multiprocessing.cpu_count()])
    z = Parallel(n_jobs=n_jobs)(delayed(parfor)(i,j) for i in range(nx) for j in range(ny))
    z = np.array(z).reshape(nx,ny)
    
    plt.contourf(x, y, z, levels=np.quantile(z,[.67,.9,.99]), **kwargs)


if __name__=='__main__':
    # set up random seed
    seed=2021
    np.random.seed(seed)
    
    # define Bayesian inverse problem
    num_traj = 1
    ode_params = {'a':0.2, 'b':0.2, 'c':5.7}
    t_init = 1000
    t_final = 1100
    time_res = 100
    obs_times = np.linspace(t_init, t_final, time_res)
    avg_traj = False
    var_out = True
    STlik = 'sep'
    rsl = Rossler(num_traj=num_traj, ode_params=ode_params, obs_times=obs_times, avg_traj=avg_traj, var_out=var_out, seed=seed, STlik=STlik)
    
    # prepare for plotting data
    para0 = rsl.misfit.true_params
    marg = [.1,.1,1]; res = 100
    grid_data = rsl.misfit.true_params.copy()
    for i,k in enumerate(grid_data):
        grid_data[k] = np.linspace(grid_data[k]-marg[i],grid_data[k]+marg[i], num=res)
    grid_data = pd.DataFrame(grid_data)
    # plot
    import time
    t_start=time.time()
    g = sns.PairGrid(grid_data, diag_sharey=False, corner=True, size=3)
    g.map_diag(plot_pdf, para0=para0, f=lambda param:rsl._get_misfit(parameter=np.log(param)))
    g.map_lower(contour, para0=para0, f=lambda param:np.exp(-rsl._get_misfit(parameter=np.log(param))), cmap='gray')
    g.savefig(os.path.join(os.getcwd(),'properties/pairpdf'+('_simple' if not STlik else '_STlik_'+STlik)+'.png'),bbox_inches='tight')
    t_end=time.time()
    print('time used: %.5f'% (t_end-t_start))