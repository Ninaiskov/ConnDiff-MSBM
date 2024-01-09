# ConnDiff-MultSBM
Code for Network Neuroscience article: "Discovering Prominent Differences in Structural and Functional Connectomes Using the Multinomial Stochastic Block Model"
The work showcase how MSBM can offer valuable insights to brain connectivity as well as many other potential applications where differences across graph-data are of interest.

Framework illustration:

<img width="500" alt="MSBM" src="https://github.com/Ninaiskov/ConnDiff-MSBM/assets/67420369/bb37427d-9c2f-4314-a682-1fde41611626">

### Data
Data used from Human Connectome Project (HCP) and synthetic data is located in data folder.

### Results
Result files 'model_sample.npy' including MAP partition matrix Z are located in results folder under respective experiment subfolder (only results files for best runs for HCP data are uploaded)

#### Examples of results for 25 clusters

Brainmap:

<img width="400" alt="brainmap" src="https://github.com/Ninaiskov/ConnDiff-MSBM/assets/67420369/5ee9ac9e-4a01-4a11-8325-cb6816656be3">

Connectivity map (red = functional conn., blue = structural conn.)

<img width="494" alt="connmap" src="https://github.com/Ninaiskov/ConnDiff-MSBM/assets/67420369/f3ef7675-6d0a-43f8-b2fd-84c6ee6ed744">


### Scripts
- main.py: Main script for defining parameters and running model
- model.py: Multinomial Stochastic Block Model (MSBM) class with Gibbs sampling inference
- createGraphs.m: Generate adjacency matrices (graphs) from dMRI (structural) and fMRI (functional) images
- get_Glassergraphs.py: Generate adjacency matrices (graphs) in Glasser atlas resolution
- get_nlink.py: Compute number of links between each cluster pair (cluster-link density)
- helper_functions.py: Helper functions (plotting functions etc.)
- run_mri_batchjobs.sh: Submit multiple batchjobs (MRI data experiments)
- run_syn_batchjobs.sh: Submit multiple batchjobs (synthetic data experiments)
- submit_big.sh: Submit single batchjobs to BIG cluster
- submit_hpc.sh: Submit single batchjobs to HPC cluster
- visualize.ipynb: Visualize data and model outputs
- speciale.yml: Conda environment
