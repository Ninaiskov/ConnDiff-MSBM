from article_helper_functions import *
import os

os.environ["OMP_NUM_THREADS"] = "5"  # set number of threads to 5

#print('Computing dmri graph')
dmri_filenames = ['dmri_sparse2.npz','dmri_sparse3.npz','dmri_sparse4.npz','dmri_sparse5.npz'] # 'dmri_sparse1.npz'
fmri_filenames = ['fmri_sparse2.npz','fmri_sparse3.npz','fmri_sparse4.npz','fmri_sparse5.npz'] # 'fmri_sparse1.npz'

for i in range(len(dmri_filenames)):
    Glasser_A_dmri = compute_Glasser_A(dmri_filenames[i])
    Glasser_A_fmri = compute_Glasser_A(fmri_filenames[i])