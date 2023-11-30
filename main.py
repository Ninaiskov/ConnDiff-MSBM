import os
import argparse
import time 
from datetime import datetime
import numpy as np
from model import MultinomialSBM

def main(config):
    # initiate results folder and log.txt file
    exp_name = config.dataset+'_'+str(datetime.now())
    config.save_dir = os.path.join(config.main_dir, 'results/'+config.dataset+'/'+exp_name)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    
    # making sure parameters make sense wrt. other parameters
    if config.model_type == 'parametric':
        config.splitmerge = False
        config.threshold_annealing = False
    if config.threshold_annealing:
        config.maxiter_gibbs = 400
        config.use_convergence_criteria = False
        
    config.use_convergence_criteria = False # TESTING
    config.threshold_annealing = False # TESTING
    print(config)
        
    # log file with specifications for experiment:
    if config.dataset == 'hcp':
        with open(os.path.join(config.save_dir, 'log.txt'), 'w') as f:
            f.write(f"dataset: {config.dataset}\n")
            f.write(f"exp_name: {exp_name}\n")
            f.write(f"model_type: {config.model_type}\n")
            f.write(f"splitmerge: {config.splitmerge}\n")
            f.write(f"noc: {config.noc}\n")
            f.write(f"maxiter_gibbs: {config.maxiter_gibbs}\n")
            f.write(f"maxiter_eta0: {config.maxiter_eta0}\n")
            f.write(f"maxiter_alpha: {config.maxiter_alpha}\n")
            #f.write(f"total_time_min: {elapsed_time}\n")
    elif config.dataset == 'synthetic':
        with open(os.path.join(config.save_dir, 'log.txt'), 'w') as f:
            f.write(f"dataset: {config.dataset}\n")
            f.write(f"exp_name: {exp_name}\n")
            f.write(f"K: {config.K}\n")
            f.write(f"S1: {config.S1}\n")
            f.write(f"S2: {config.S2}\n")
            f.write(f"Nc_type: {config.Nc_type}\n")
            f.write(f"alpha: {config.alpha}\n")
            f.write(f"model_type: {config.model_type}\n")
            f.write(f"splitmerge: {config.splitmerge}\n")
            f.write(f"noc: {config.noc}\n")
            f.write(f"maxiter_gibbs: {config.maxiter_gibbs}\n")
            f.write(f"maxiter_eta0: {config.maxiter_eta0}\n")
            f.write(f"maxiter_alpha: {config.maxiter_alpha}\n")
    else: 
        print('Unknown dataset. Please choose between synthetic or hcp.')
    
    start_time = time.time()
    
    #%% Run code
    print('Using ' + config.dataset + ' dataset')
    model = MultinomialSBM(config)
    
    model.train()
    # SAVE MODEL OUTPUTS (final)
    np.save(os.path.join(config.save_dir,'model_sample'+str(config.maxiter_gibbs)+'.npy'), model.sample)

    elapsed_time = (time.time() - start_time) /60
    print('total_time_min:', elapsed_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data configuration.
    parser.add_argument('--dataset', type=str, default='synthetic', help='dataset name (synthetic, hcp, decnef)')
        # Synthetic data configuration. 
    parser.add_argument('--K', type=int, default=5, help='number of clusters (synthetic data)')
    parser.add_argument('--S1', type=int, default=5, help='number of graphs of type 1 (synthetic data)')
    parser.add_argument('--S2', type=int, default=5, help='number of graphs of type 2 (synthetic data)')
    parser.add_argument('--Nc_type', type=str, default='unbalanced', help='balanced or unbalanced no. of nodes in each cluster')
    parser.add_argument('--alpha', type=float, default=0, help='scaling parameter for similiarty between eta_p1 and eta_p2 (used for article synthetic data)') # only used in article

    # Model configuration.
    parser.add_argument('--model_type', type=str, default='parametric', help='model type (nonparametric/parametric)')
    parser.add_argument('--noc', type=int, default=50, help='intial number of clusters')
    parser.add_argument('--splitmerge', type=bool, default=True, help='use splitmerge for nonparametric model (True/False)')
    
    # Training configuration.
    parser.add_argument('--maxiter_gibbs', type=int, default=100, help='max number of gibbs iterations')
    parser.add_argument('--maxiter_eta0', type=int, default=10, help='max number of MH iterations for sampling eta0')
    parser.add_argument('--maxiter_alpha', type=int, default=100, help='max number of MH iterations for sampling alpha')
    parser.add_argument('--maxiter_splitmerge', type=int, default=10, help='max number of splitmerge iterations')
    parser.add_argument('--unit_test', type=bool, default=False, help='perform unit test (True/False)')
    parser.add_argument('--use_convergence_criteria', type=bool, default=True, help='use convergence criteria (True/False). If True, the algorithm stops when the convergence criteria is met')
    
    # Miscellaneous.
    parser.add_argument('--main_dir', type=str, default='/work3/s174162/speciale/', help='main directory')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to save results')
    parser.add_argument('--disp', type=bool, default=True, help='display iteration results (True/False)')
    parser.add_argument('--sample_step', type=int, default=1, help='number of iterations between each logged sample')
    parser.add_argument('--save_step', type=int, default=10, help='number of iterations between each saved sample (temporay results files)')

    config = parser.parse_args()
    main(config)
