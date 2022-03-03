"""
Get relative error of mean (rem) in EnK algorithms for uncertainty field u in chaotic inverse problem.
Shiwei Lan @ ASU, 2022
----------------------
"""

import os,pickle
import numpy as np
import dolfin as df


# seed=2021
# truth
true_param = list({'h':1.0, 'F':10, 'logc':np.log(10),'b':10}.values())

# algorithms and settings
algs=('EKI','EKS')
num_algs=len(algs)
lik_mdls=('simple','STlik')
num_mdls=len(lik_mdls)
ensbl_szs=[50,100,200,500,1000]
num_ensbls=len(ensbl_szs)
# store results
rem_m=np.zeros((num_mdls,num_algs,num_ensbls))
rem_s=np.zeros((num_mdls,num_algs,num_ensbls))
# obtain estimates
folder = './analysis'
for m in range(num_mdls):
    print('Processing '+lik_mdls[m]+' likelihood model...\n')
    # fld_m = folder+('_fixedhyper/' if m==0 else '/')+lik_mdls[m]
    fld_m = folder+'/'+lik_mdls[m]
    # preparation for estimates
    pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
    for i in range(num_algs):
        for j in range(num_ensbls):
            print('Getting estimates for '+algs[i]+' algorithm with '+str(ensbl_szs[j])+' ensembles...')
            # calculate posterior estimates
            rems=[]
            num_read=0
            for f_i in pckl_files:
                if '_'+algs[i]+'_ensbl'+str(ensbl_szs[j])+'_' in f_i:
                    try:
                        f=open(os.path.join(fld_m,f_i),'rb')
                        f_read=pickle.load(f)
                        u_est,err=f_read[:2]
                        u_est=u_est[err!=0]; err=err[err!=0]
                        param_est=u_est[np.argmin(err)-1]
                        rems.append(np.linalg.norm(np.exp(param_est)-true_param)/np.linalg.norm(true_param))
                        num_read+=1
                        f.close()
                        print(f_i+' has been read!')
                    except:
                        pass
            print('%d experiment(s) have been processed for %s algorithm with %d ensembles for %s likelihood model.' % (num_read, algs[i], ensbl_szs[j], lik_mdls[m]))
            if num_read>0:
                rems = np.stack(rems)
                rem_m[m,i,j] = np.median(rems)
                rem_s[m,i,j] = rems.std()
# save
import pandas as pd
# rem_m = pd.DataFrame(data=rem_m,index=lik_mdls,columns=alg_names[:num_algs])
# rem_s = pd.DataFrame(data=rem_s,index=lik_mdls,columns=alg_names[:num_algs])
# rem_m.to_csv(os.path.join(folder,'REM-mean.csv'),columns=alg_names[:num_algs])
# rem_s.to_csv(os.path.join(folder,'REM-std.csv'),columns=alg_names[:num_algs])
rem_m = pd.DataFrame(data=rem_m.reshape((-1,num_ensbls)),index=[m+'_'+i for m in lik_mdls for i in algs],columns=['J='+str(j) for j in ensbl_szs])
rem_s = pd.DataFrame(data=rem_s.reshape((-1,num_ensbls)),index=[m+'_'+i for m in lik_mdls for i in algs],columns=['J='+str(j) for j in ensbl_szs])
rem_m.to_csv(os.path.join(folder,'REM-mean.csv'),columns=['J='+str(j) for j in ensbl_szs])
rem_s.to_csv(os.path.join(folder,'REM-std.csv'),columns=['J='+str(j) for j in ensbl_szs])