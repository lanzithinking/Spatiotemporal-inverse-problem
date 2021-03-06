# Spatiotemporal-inverse-problem

## Bayesian spatiotemporal modeling for inverse problems (B-STIP)
[https://arxiv.org/abs/2204.10929](https://arxiv.org/abs/2204.10929)

### software preparation
* [**FEniCS**](https://fenicsproject.org) Go to this [webpage](https://fenicsproject.org/download/) for installation.

* [**hIPPYlib**](https://hippylib.github.io) Go to this [webpage](https://hippylib.readthedocs.io/en/3.0.0/installation.html) for installation.

* It is **recommended** to use *Conda* to install `FEniCS` and *pip* to install `hIPPYlib`. Create a `FEniCS` environment. 
Load `FEniCS` environment in terminal session and include `hIPPYlib` in PYTHONPATH for that session.
Alternatively, install `hIPPYlib` directly in the `FEniCS` environment.


### package structure
* **adv-diff** contains files for advection-diffusion inverse problem (requires `FEniCS` and `hIPPYlib`).
	* `run_advdiff_geoinfMC.py` to collect samples using geometric infinite-dimensional MCMC algorithms (*pCN*, *inf-MALA* and *inf-HMC*).
	* `plot_mcmc_estimates_comparelik.py` to plot MCMC estimates (mean and standard deviation) for different likelihood models.
	* `get_mcmc_rem_comparelik.py` to generate table comparing relative errors of posterior mean between different likelihood models.
	* `get_prederr_comparelik.py` to generate table comparing prediction errors of forward outputs between different likelihood models.
	* `plot_predictions_comparelik.py` to plot forward prediction at selective locations and compare the truth covering rate of credible bands.
* **chaotic dynamical inverse problems** contain 3 chaotic dynamics **Lorenz(63)**, **Rossler** and **Chen** each having:
	* `run_XXX_EnK.py` to collect ensembles using ensemble Kalman (*EnK*) methods (*EKI* and *EKS*).
	* `run_XXX_EnK_spinavgs.py` to run EnK algorithms by varying spin-up length `t<sub>0</sub>` and observation window size `T`.
	* `get_enk_rem_comparelik.py` to generate table comparing relative errors of posterior mean between different likelihood models.
	* `plot_enk_rem_comparelik.py` to plot relative errors of mean by EnK between different likelihood models.
	* `plot_enk_rem_spinavgs.py` to plot relative errors of mean for different spin-up lengths and window sizes between different likelihood models.
	* `plot_predictions_comparelik.py` to plot forward predictions.
	* `prep_traindata.py` to extract training data for NN emulator from EnK outputs.
	* `run_XXX_einfGMC.py` to run MCMC algorithms based on NN emulators.
	* `plot_postdist.py` to plot marginal and pairwise posterior densities based on emulative MCMC samples.
* **optimizer** contains ensemble Kalman algorithms as optimization (EKI) or approximate sampling (EKS) methods.
	* `EnK.py`: Ensemble Kalman algorithms
* **sampler** contains different MCMC algorithms
	* `geoinfMC.py`: infinite-dimensional Geometric Monte Carlo.
	* `geoinfMC_dolfin.py`: infinite-dimensional Geometric Monte Carlo (working with `dolfin` in `FEniCS`).
* **util** contains utility functions supplementary to dolfin package in `FEniCS`.
	* `stgp`: spatiotemporal Gaussian process models.
* **nn** contains neural network definitions:
	* `dnn`: densely connected neural network.
	* `dnn_rnn`: DNN-RNN (recurrent neural network) type of network.

* Simple likelihood method refers to either static model (adv-diff) or time-averaged approach (chaotic dynamics).