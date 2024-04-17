# ConnDiff-MultSBM
Code for article: "Discovering Prominent Differences in Structural and Functional Connectomes Using the Multinomial Stochastic Block Model"

This work showcase how MSBM can offer valuable insights to brain connectivity as well as many other potential applications where differences across graph-data are of interest.

#### Framework illustration

<img width="857" alt="MSBM" src="https://github.com/Ninaiskov/ConnDiff-MSBM/assets/67420369/1469f079-f942-4eed-acb3-ed2b733903c6">


### Data
Data used is open-access neuroimaging dataset (fMRI and dMRI scans) from Human Connectome Project (HCP) and synthetically generated data

### Results
Result files 'model_sampleK.npy' containing MAP partition matrix Z (N nodes partitioned into K clusters) are located in results/hcp_best (only results files for best runs for HCP data are uploaded).

#### Examples of HCP results for 25 clusters

Brainmap:

<img width="300" alt="brainmap" src="https://github.com/Ninaiskov/ConnDiff-MSBM/assets/67420369/567cb112-d386-4178-950f-91693cca2f94">



Connectivity map (Probability of link between clusters, where red = functional conn., blue = structural conn.)

<img width="600" alt="K25" src="https://github.com/Ninaiskov/ConnDiff-MSBM/assets/67420369/d6995391-8f3a-4541-9992-63aee014b3e8">






### Files
- main.py: Main script for defining parameters and running model
- model.py: Multinomial Stochastic Block Model (MSBM) class with Gibbs sampling inference
- createGraphs.m: Generate adjacency matrices (graphs) from dMRI (structural) and fMRI (functional) images
- get_Glassergraphs.py: Generate adjacency matrices (graphs) in Glasser atlas resolution
- helper_functions.py: Helper functions (plotting functions etc.)
- visualize.ipynb: Visualize data and model outputs
- requirements.txt: Python package requirements


### Setup and run

1. Clone the repository
``` 
git clone https://github.com/Ninaiskov/ConnDiff-MSBM.git
``` 

2. Create a conda environment and install required packages
``` 
conda create --name <env_name> --file requirements.txt
```

3. Activate the environment
```
conda activate <env_name>
```

4. Run the main.py script to run the model
```
python main.py
```
