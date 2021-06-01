"""
Analyze MCMC samples
Shiwei Lan @ U of Warwick, 2016; @ Caltech, Sept. 2016
------------------------------------------------------
Modified for DREAM December 2020 @ ASU
"""

import os
import dolfin as df
import numpy as np

import sys
sys.path.append( "../" )
from util.bayesianStats import effectiveSampleSize as ess
from joblib import Parallel, delayed

# def restore_each_sample(f,samp_f,s):
#     f.read(samp_f,'sample_{0}'.format(s))
#     return samp_f.vector()

def restore_sample(mpi_comm,V,dir_name,f_name,num_samp):
    f=df.HDF5File(mpi_comm,os.path.join(dir_name,f_name),"r")
    samp_f=df.Function(V,name="parameter")
    samp=np.zeros((num_samp,V.dim()))
    prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
    for s in range(num_samp):
        f.read(samp_f,'sample_{0}'.format(s))
        samp[s,]=samp_f.vector()
        if s+1 in prog:
            print('{0:.0f}% samples have been restored.'.format(np.float(s+1)/num_samp*100))
#     f_read=lambda s: restore_each_sample(f,samp_f,s)
#     samp=Parallel(n_jobs=4)(delayed(f_read)(i) for i in range(num_samp))
    f.close()
    print(f_name+' has been read!')
    return samp

def get_ESS(samp):
    ESS=Parallel(n_jobs=4)(map(delayed(ess), np.transpose(samp)))
    return ESS

if __name__ == '__main__':
    from advdiff import advdiff
    # define the inverse problem
    seed=2020
    np.random.seed(seed)
    meshsz = (61,61)
    eldeg = 1
    gamma = 2.; delta = 10.
    rel_noise = .5
    nref = 1
    adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
    adif.prior.V=adif.prior.Vh
    adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
     # define the latent (coarser) inverse problem
    meshsz_latent = (21,21)
    adif_latent = advdiff(mesh=meshsz_latent, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
    adif_latent.prior.V=adif_latent.prior.Vh
    # algorithms
    algs=('pCN','infMALA','infHMC','epCN','einfMALA','einfHMC','DREAMpCN','DREAMinfMALA','DREAMinfHMC')
    alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','e-pCN','e-$\infty$-MALA','e-$\infty$-HMC','DREAM-pCN','DREAM-$\infty$-MALA','DREAM-$\infty$-HMC')
    num_algs=len(algs)
    # preparation for estimates
    folder = './analysis_eldeg'+str(eldeg)
    fnames=[f for f in os.listdir(folder) if f.endswith('.h5')]
    num_samp=5000
    
    # calculate ESS's
    ESS=(np.zeros(adif.prior.V.dim()),)*6+(np.zeros(adif_latent.prior.V.dim()),)*3
    sumry_ESS=np.zeros((num_algs,3))
    found=np.zeros(num_algs,dtype=bool)
    for a in range(num_algs):
        print('Working on '+algs[a]+' algorithm...')
        _ESS=[]
        bip=adif_latent if 'DREAM' in algs[a] else adif
        # samples
        for f_i in fnames:
            if '_'+algs[a]+'_' in f_i:
                try:
                    samp=restore_sample(bip.mpi_comm,bip.prior.V,folder,f_i,num_samp)
                    _ESS_i=get_ESS(samp)
                    _ESS.append(_ESS_i)
                    found[a]=True
                except Exception as err:
                    print(err)
                    pass
        if found[a]:
            ESS_a=np.mean(_ESS,axis=0)
            sumry_ESS[a]=[ESS_a.min(),np.median(ESS_a),ESS_a.max()]
            # select some dimensions for plot
            samp_fname=os.path.join(folder,algs[a]+'_selected_samples.txt')
            if not os.path.isfile(samp_fname):
                par_dim=samp.shape[1]
                select_indices=np.ceil(par_dim*(np.linspace(0,1,6)));select_indices[-1]=par_dim-1;select_indices=np.int_(select_indices)
                select_samples=np.vstack((select_indices,samp[:,select_indices]))
                np.savetxt(samp_fname,select_samples,delimiter=',')
    
    # save the result to file
    if any(found):
        found_idx=np.where(found)[0]
        sumry_ESS=np.hstack((np.array(algs)[found_idx,None],sumry_ESS[found_idx,:]))
        np.savetxt(os.path.join(folder,'sumry_ESS.txt'),sumry_ESS,fmt="%s",delimiter=',')
    