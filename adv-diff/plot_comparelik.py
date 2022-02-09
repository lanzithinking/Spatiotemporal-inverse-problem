"""
Plot estimates of uncertainty field u in Advection-Diffusion inverse problem.
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Modified for STIP August 2021 @ ASU
"""

import os,pickle
import numpy as np
import dolfin as df
# import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mp

from advdiff import advdiff
import sys
sys.path.append( "../" )
from util.dolfin_gadget import *
from util.multivector import *


seed=2020
# define the inverse problem
meshsz = (61,61)
eldeg = 1
gamma = 2.; delta = 10.
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
# get the true parameter (initial condition)
ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
true_param = df.interpolate(ic_expr, adif.prior.V).vector()

# plot
folder = './analysis_eldeg'+str(eldeg)
plt.rcParams['image.cmap'] = 'jet'
fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(13,4))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot truth
        df.plot(vec2fun(true_param,adif.prior.V))
        ax.set_title('Truth',fontsize=18)
    else:
        # plot MAP
        try:
            # # file='MAP'+{1:'0',2:''}[i]+'.xdmf'
            # if i==1:
            #     f=df.XDMFFile(adif.mpi_comm, os.path.join('/Users/slan/Projects/BUQae/code/ad_diff/properties','MAP.xdmf'))
            # elif i==2:
            # f=df.XDMFFile(adif.mpi_comm, os.path.join('../../../BUQae/code/ad_diff' if i==1 else os.getcwd(),'properties','MAP.xdmf'))
            f=df.XDMFFile(adif.mpi_comm, os.path.join(os.getcwd(),'properties','MAP_static.xdmf' if i==1 else 'MAP_static.xdmf'))
            MAP=df.Function(adif.prior.V,name="MAP")
            f.read_checkpoint(MAP,'m',0)
            f.close()
            sub_figs[i]=df.plot(MAP)
            # fig.colorbar(sub_figs[i],ax=ax)
            ax.set_title('MAP ('+{1:'static',2:'STGP'}[i]+' model)',fontsize=18)
        except Exception as e:
            print(e)
            pass
    ax.set_aspect('auto')
    plt.axis([0, 1, 0, 1])
# set color bar
# cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(sub_fig, cax=cax, **kw)
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/comparelik.png',bbox_inches='tight')
# plt.show()
