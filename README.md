# ConnDiff-MultSBM
Code for paper: Uncovering Prominent Differences in Structural and Functional Connectomes Using the Multinomial Stochastic Block Model

### Data
Data used from Human Connectome Project (HCP) and synthetic data is located in data folder.

### Results
Result files 'model_sample.npy' including MAP partition Z are located in results folder under respective experiment subfolder.

### Scripts
- main.py: Main script for defining parameters and running model
- model.py: Multinomial Stochastic Block Model (mSBM) class with Gibbs sampling inference
- createGraphs.m: Generate adjacency matrices (graphs) from fMRI and dMRI images  
- helper_functions.py: Helper functions
- run_mri_batchjobs.sh: Submit multiple batchjobs (MRI data experiments)
- run_syn_batchjobs.sh: Submit multiple batchjobs (synthetic data experiments)
- submit_big.sh: Submit single batchjobs to BIG cluster
- submit_hpc.sh: Submit single batchjobs to HPC cluster
- visualize.ipynb: Visualize data and model outputs
- speciale.yml: Conda environment
