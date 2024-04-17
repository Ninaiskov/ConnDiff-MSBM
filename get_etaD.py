import os
import numpy as np
from helper_functions import compute_etaD

# Script for computing cluster-link density matrices (etaD) for each cluster assignment (z) from the MAP estimate

os.environ["OMP_NUM_THREADS"] = "5"  # set number of threads

main_dir = os.getcwd()
dataset = '100_extra_func' #'100_extra_struc', 'hcp'
data_path = os.path.join(main_dir, 'data', dataset)
if dataset == '100_extra_func':
        extra_conn_idx = np.load(os.path.join(main_dir, 'data/extra_conn_idx.npy')).tolist()
        filename_list = ['graph_fMRI_'+idx for idx in extra_conn_idx]
elif dataset == '100_extra_struc':
    extra_conn_idx = np.load(os.path.join(main_dir, 'data/extra_conn_idx.npy')).tolist()
    filename_list = ['graph_dMRI_'+idx for idx in extra_conn_idx]
elif dataset == 'hcp':
    filename_list = [file for file in os.listdir(data_path) if file.endswith('.npz')]
else:
    print('Unknown dataset specified. Use either 100_extra_func or 100_extra_struc')
'''
# get experiment dataframe
top_dir = os.path.join('results', dataset)
df_exp = get_exp_overview(top_dir)
# sort dataframe
df_sorted = df_exp[df_exp.maxiter_gibbs == 100].sort_values(by='noc').reset_index(drop=True)
'''
# choose experiment
for noc in [2,3,4,25,50,100]: # 2, 3, 4, 25, 50, 100
    '''
    exp_folder = get_best_run(df=df_sorted, noc=noc) # experiment visualized is the one with highest MAP estimate across random runs

    # extract results
    if noc < 100:
        maxiter_gibbs = 100
    else: # also including where noc=100 (they only ran 30 iterations)
        maxiter_gibbs = 30
    model_sample = np.load(os.path.join(main_dir, 'results/hcp/'+exp_folder+'/model_sample'+str(maxiter_gibbs)+'.npy'), allow_pickle=True).item()
    '''
    model_sample = np.load(os.path.join(main_dir, 'results/hcp_best/model_sample'+str(noc)+'.npy'), allow_pickle=True).item()
    model_sample.keys()

    MAP_sample = model_sample['MAP']
    Z = MAP_sample['Z']
    z = np.argmax(Z, axis=0)+1 # cluster assignments
    etaD_list = []
    for filename in filename_list:
        etaD = compute_etaD(data_path=data_path, filename=filename, z=z)
        etaD_list.append(etaD)
    np.save(os.path.join(data_path,'etaD_'+str(noc)+'.npy'),np.stack(etaD_list).T)