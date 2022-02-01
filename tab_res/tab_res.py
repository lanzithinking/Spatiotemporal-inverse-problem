"""generate latex table results.

Shiwei Lan @ U of Warwick, 2016
"""

import numpy as np
import pandas as pd

# algs=('pCN','infMALA','infHMC')#,'epCN','einfMALA','einfHMC','DREAMpCN','DREAMinfMALA','DREAMinfHMC','DRinfmHMC')
# alg_names=('pCN','$\infty$-MALA','$\infty$-HMC','e-pCN','e-$\infty$-MALA','e-$\infty$-HMC','DREAM-pCN','DREAM-$\infty$-MALA','DREAM-$\infty$-HMC','DR-$\infty$-HMC')
# num_algs=len(algs)
# lik_mdls=('simple','STlik')
# num_mdls=len(lik_mdls)

# advection-diffusion inverse problem
# relative error of mean in posterior estimates
try:
    rem_m = pd.read_csv('../adv-diff/analysis_eldeg1/REM-mean.csv')
    rem_s = pd.read_csv('../adv-diff/analysis_eldeg1/REM-std.csv')
    sumry = rem_m.round(decimals=2)
    for i in range(sumry.shape[0]):
        for j in range(1,sumry.shape[1]):
            sumry.iat[i,j] = str(sumry.iat[i,j])+' ('+np.array2string(rem_s.iat[i,j],precision=3)+')'
    # sumry.columns = alg_names[:num_algs]
    sumry.columns.values[0] = 'Models'
    sumry_tab = sumry.to_latex(index=False, escape=False)
    # print(sumry_tab)
    f = open('../adv-diff/analysis_eldeg1/comparelik.tex', 'w')
    f.write(sumry_tab)
    f.close()
except Exception as e:
    print(e)


# relative error in prediction
try:
    err_m = pd.read_csv('../adv-diff/analysis_eldeg1/prederr-mean.csv')
    err_s = pd.read_csv('../adv-diff/analysis_eldeg1/prederr-std.csv')
    sumry = err_m.round(decimals=2)
    for i in range(sumry.shape[0]):
        for j in range(1,sumry.shape[1]):
            sumry.iat[i,j] = str(sumry.iat[i,j])+' ('+np.array2string(err_s.iat[i,j],precision=3)+')'
    # sumry.columns = alg_names[:num_algs]
    sumry.columns.values[0] = 'Models'
    sumry_tab = sumry.to_latex(index=False, escape=False)
    # print(sumry_tab)
    f = open('../adv-diff/analysis_eldeg1/pred_comparelik.tex', 'w')
    f.write(sumry_tab)
    f.close()
except Exception as e:
    print(e)

# algs=('EKI','EKS')
# num_algs=len(algs)

# Rossler dynamical inverse problem
# relative error of mean in posterior estimates
try:
    rem_m = pd.read_csv('../Rossler/analysis/REM-mean.csv')
    rem_s = pd.read_csv('../Rossler/analysis/REM-std.csv')
    sumry = rem_m
    fmt = lambda x: "%.2e" % x if abs(x)<.01 else "%.2f" % x
    for i in range(sumry.shape[0]):
        for j in range(1,sumry.shape[1]):
            # sumry.iat[i,j] = str("%.2g" % sumry.iat[i,j])+' ('+np.array2string(rem_s.iat[i,j],formatter={'float_kind':lambda x: "%.2g" % x})+')'
            sumry.iat[i,j] = str(fmt(sumry.iat[i,j]))+' ('+str(fmt(rem_s.iat[i,j]))+')'
    # sumry.columns = alg_names[:num_algs]
    sumry.columns.values[0] = 'Model-Algorithms'
    sumry_tab = sumry.to_latex(index=False, escape=False)
    # print(sumry_tab)
    f = open('../Rossler/analysis/comparelik.tex', 'w')
    f.write(sumry_tab)
    f.close()
except Exception as e:
    print(e)