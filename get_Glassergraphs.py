from helper_functions import *
import os

os.environ["OMP_NUM_THREADS"] = "5"  # set number of threads to 5

# choose experiment
dataset = 'hcp'
top_dir = os.path.join('results', dataset)
main_dir = os.getcwd()
df_exp = get_exp_overview(top_dir)
# sort dataframe
df_sorted = df_exp[df_exp.maxiter_gibbs == 100].sort_values(by='noc').reset_index(drop=True)

noc_list = [2,3,4,25,50,100] # 2, 3, 4, 25, 50, 100
# computing node labels using (cluster assignments) from partition
filenames = ['fmri_sparse1.npz', 'fmri_sparse2.npz','fmri_sparse3.npz','fmri_sparse4.npz','fmri_sparse5.npz',
            'dmri_sparse1.npz', 'dmri_sparse2.npz','dmri_sparse3.npz','dmri_sparse4.npz','dmri_sparse5.npz']
data_path = os.path.join(main_dir, 'data','hcp')
for noc in noc_list:
    exp_folder = get_best_run(df=df_sorted, noc=noc) # experiment visualized is the one with highest MAP estimate across random runs

    # extract results
    if noc < 100:
        maxiter_gibbs = 100
    else: # also including where noc=100 (they only ran 30 iterations)
        maxiter_gibbs = 30
    model_sample = np.load(os.path.join(main_dir, 'results/hcp/'+exp_folder+'/model_sample'+str(maxiter_gibbs)+'.npy'), allow_pickle=True).item()

    MAP_sample = model_sample['MAP']
    Z = MAP_sample['Z'] # partition matrix
    z = np.argmax(Z, axis=0)+1 # cluster assignments
    etaD_list = []
    for filename in filenames:
        etaD = compute_etaD(data_path=data_path, filename=filename, z=z)
        etaD_list.append(etaD)
    np.save(os.path.join(data_path,'etaD_'+str(noc)+'.npy'),np.stack(etaD_list).T)
