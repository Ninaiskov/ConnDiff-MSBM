import os
import numpy as np
from numba import njit, prange
from scipy.sparse import load_npz, triu
from helper_functions import get_exp_overview, get_best_run

# Script for computing number of links between each cluster pair (cluster-link density)

os.environ["OMP_NUM_THREADS"] = "5"  # set number of threads

main_dir = '/work3/s174162/speciale'
dataset = 'hcp'
data_path = os.path.join(main_dir, 'data/'+dataset)

## numba code for matrix multiplication between parallel csr sparse matrix A and dense matrix B
# wrapper with initialization of result array
def spdenmatmul(A, B):
    out = np.zeros((A.shape[0],B.shape[1]),B.dtype)
    spmatmul(A.data, A.indptr, A.indices, B, out)
    return out

# parallel sparse matrix multiplication, note that the array is incremented
# multiple runs will there sum up not reseting the array in-between
# possibly types can be added for potential extra speedup
@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def spmatmul(A, iA, jA, B, out):
    for i in prange(out.shape[0]):
        for j in range(iA[i], iA[i+1]):
            for k in range(out.shape[1]):
                out[i, k] += A[j] * B[jA[j], k]
                
def load_data(main_dir, dataset):
    print('loading data..')
    data_path = os.path.join(main_dir, 'data/'+dataset)
    filename_list = ['fmri_sparse1.npz', 'fmri_sparse2.npz', 'fmri_sparse3.npz', 'fmri_sparse4.npz', 'fmri_sparse5.npz', 
                    'dmri_sparse1.npz', 'dmri_sparse2.npz', 'dmri_sparse3.npz', 'dmri_sparse4.npz', 'dmri_sparse5.npz']
    A = []
    for filename in filename_list:
        graph = load_npz(os.path.join(data_path, filename)).astype(dtype=np.int32) # single graph
        graph_sym = triu(graph,1)+triu(graph,1).T
        A.append(graph_sym)
    return A

def compute_n_link(Z, A, eta0):
    print('computing n_link..')
    S = len(eta0)
    n_link = np.stack([Z @ spdenmatmul(As, Z.T) for As in A], axis=2) # used for list of scipy sparse csr matrix (NEW numba version)
    n_link = np.stack([n_link[:, :, s] - 0.5 * np.diag(np.diag(n_link[:, :, s])) + eta0[s] for s in range(S)], axis=2) # old line that works (can probably be optimized)
    return n_link


# get experiment dataframe
top_dir = os.path.join('results', dataset)
df_exp = get_exp_overview(top_dir)
# sort dataframe
df_sorted = df_exp[df_exp.maxiter_gibbs == 100].sort_values(by='noc').reset_index(drop=True)

# choose experiment
for noc in [4]: #noc = 25 # 2, 3, 4, 25, 50, 100
    exp_folder = get_best_run(df=df_sorted, noc=noc) # experiment visualized is the one with highest MAP estimate across random runs

    # extract results
    if noc < 100:
        maxiter_gibbs = 100
    else: # also including where noc=100 (they only ran 30 iterations)
        maxiter_gibbs = 30
    model_sample = np.load(os.path.join(main_dir, 'results/hcp/'+exp_folder+'/model_sample'+str(maxiter_gibbs)+'.npy'), allow_pickle=True).item()
    model_sample.keys()

    MAP_sample = model_sample['MAP']
    Z = MAP_sample['Z']
    eta0 = MAP_sample['eta0']
    A = load_data(main_dir=main_dir, dataset=dataset)
    n_link = compute_n_link(Z=Z, A=A, eta0=eta0)
    np.save(os.path.join(data_path,'n_link'+str(noc)+'.npy'), n_link)
