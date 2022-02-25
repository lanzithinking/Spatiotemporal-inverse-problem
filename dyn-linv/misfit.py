#!/usr/bin/env python
"""
Class definition of data-misfit for dynamic linear model.
---------------------------------------------------------------
Created February 15, 2022 for project of Bayesian Spatiotemporal inverse problem (B-STIP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2021, The Bayesian STIP project"
__credits__ = "Mirjeta Pasha"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import scipy.io as spio
from scipy.ndimage import convolve

import os,sys
sys.path.append( "../" )

class misfit(object):
    """
    Class definition of data-misfit function
    """
    def __init__(self, **kwargs):
        """
        Initialize data-misfit class with information of observations.
        """
        self.truth = spio.loadmat('satellite.mat')['x_true']
        self.size = self.truth.shape
        self.fltsz = kwargs.pop('fltsz',[21,21]) # filter size
        self.fltnz = kwargs.pop('fltnz',2.56) # Gaussian filter std
        self.nzlvl = kwargs.pop('nzlvl',0.02) # noise level
        # get observations
        self.obs, self.nzvar = self.get_obs(**kwargs)
        
    def _proj(self, input, filter=None, mode='constant', direction='fwd', **kwargs):
        """
        Project the image
        """
        if np.ndim(input)!=2: input=input.reshape(self.size)
        if filter is None: filter = self._filter()
        if direction=='bkd': filter = np.flipud(np.fliplr(filter)) 
        proj_img = convolve(input, weights=filter, mode=mode, **kwargs)
        return proj_img
    
    def _filter(self, fltsz=None, fltnz=None):
        """
        Gaussian filter
        """
        if fltsz is None:
            fltsz=self.fltsz
        if hasattr(fltsz, "__len__"):
            m, n = fltsz[0], fltsz[1]
        else:
            m, n = fltsz, fltsz
        if fltnz is None:
            fltnz=self.fltnz
        x = np.arange(-np.fix(n/2), np.ceil(n/2))
        y = np.arange(-np.fix(m/2), np.ceil(m/2))
        X, Y = np.meshgrid(x, y)
        wts = np.exp( -0.5* ((X**2)/(fltnz**2) + (Y**2)/(fltnz**2)) )
        wts /= wts.sum()
        return wts
    
    def observe(self, img, filter=None, **kwargs):
        """
        Observe image by adding noise
        """
        if np.ndim(img)!=2: img=img.reshape(self.size)
        if filter is None:
            filter = self._filter()
        obs_img = self._proj(img, filter, **kwargs)
        nzstd = self.nzlvl * np.linalg.norm(obs_img)/img.shape[0]
        return obs_img, nzstd**2
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='satellite_obs'
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzvar=loaded['nzvar']
            print('Observation file '+obs_file_name+' has been read!')
        except Exception as e:
            print(e); pass
            img = kwargs.pop('img',self.truth)
            filter = self._filter(kwargs.pop('fltsz',self.fltsz), kwargs.pop('fltnz',self.fltnz))
            obs, nzvar = self.observe(img, filter, **kwargs)
            obs += np.sqrt(nzvar) * np.random.RandomState(kwargs.pop('rand_seed',2021)).randn(*img.shape)
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzvar=nzvar)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzvar
    
    def cost(self, u):
        """
        Evaluate misfit function for given image (vector) u.
        """
        obs = self._proj(u)
        dif_obs = obs-self.obs
        val = 0.5*np.sum(dif_obs**2)/self.nzvar
        return val
    
    def grad(self, u):
        """
        Compute the gradient of misfit
        """
        obs = self._proj(u)
        dif_obs = obs-self.obs
        g = self._proj(dif_obs,direction='bkd').flatten()/self.nzvar
        return g
    
    def plot_data(self):
        """
        Plot the data information.
        """
        import matplotlib.pyplot as plt
        plt.set_cmap('Greys')
        # from util import matplot4dolfin
        # matplot=matplot4dolfin()
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
        for i,ax in enumerate(axes.flat):
            img = {0:self.truth,1:self.obs}[i]
            plt.axes(ax)
            plt.imshow(img, origin='lower',extent=[0, 1, 0, 1])
            ax.set_title({0:'Truth',1:'Observation'}[i],fontsize=16)
            ax.set_aspect('auto')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        return fig
    
if __name__ == '__main__':
    np.random.seed(2021)
    # define the misfit
    msft = misfit()
    # test
    nll=msft.cost(msft.obs)
    grad=msft.grad(msft.obs)
    h=1e-6
    gradv_fd=(msft.cost(msft.obs+h*msft.truth)-nll)/h
    gradv=grad.dot(msft.truth.flatten())
    rdiff_gradv=np.abs(gradv_fd-gradv)/np.linalg.norm(msft.truth)
    print('Relative difference of gradients in a direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    # # plot
    # import matplotlib.pyplot as plt
    # fig=msft.plot_data()
    # # fig.tight_layout()
    # fig.savefig('./properties/truth_obs.png',bbox_inches='tight')
    # plt.show()