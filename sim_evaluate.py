"""Calculate inferential utility metrics of original and synthetic datasets"""

import os
import pandas as pd
import numpy as np
import itertools
from itertools import product
from time import time
from utils.eval import estimate_estimator, CI_coverage
from utils.disease import ground_truth

def load_data(n_samples: int, n_runs: int, sim_dir: str):
    """
    Load source files containing original and synthetic datasets
    """    
    
    # Create empty data structure for storage
    data = {}
    for n_sample in n_samples:
        data[f'n_{n_sample}'] = {}
        for i in range(n_runs):
            data[f'n_{n_sample}'][f'run_{i}'] = {}

    # Load datasets
    data_dir = [os.path.join(root, name) for root, dirs, files in os.walk(sim_dir) for name in files]
    for dir in data_dir:
        
        if '.ipynb_checkpoint' in dir:
            continue # skip dot files
        
        dir_structure = dir[len(sim_dir):].split('/') # remove sim_dir prefix from name and split directory by '/'
        try: # load datasets per n_sample and per Monte Carlo run
            data[dir_structure[0]][dir_structure[1]][dir_structure[2][:-4]] = pd.read_csv(dir, engine='pyarrow') # also remove '.csv' suffix from name
        except Exception as e:
            print(f'File {dir} not stored in \'data\' object because file has aberrant directory structure')

    # Remove empty simulations
    for n_sample in n_samples:
        for i in range(n_runs):
            if not data[f'n_{n_sample}'][f'run_{i}']:
                del data[f'n_{n_sample}'][f'run_{i}']
    
    return(data)

def create_metadata(data: pd.DataFrame, export: bool=False):
    """
    Create meta_data (or update if additional generative models have been added) per n_sample and per Monte Carlo run
    """
    
    for n_sample in data:
        
        for run in data[n_sample]:
            
            all_data = data[n_sample][run] # original and synthetic datasets
            meta_data = pd.DataFrame({'dataset_name': [name for name in all_data if name not in ['meta_data']],
                                      'n': [all_data[name].shape[0] for name in all_data if name not in ['meta_data']],
                                      'run': np.repeat(run, len([name for name in all_data if name not in ['meta_data']])),
                                      'generator': ['_'.join(name.split('_')[:-1]) for name in all_data if name not in ['meta_data']]})
            data[n_sample][run]['meta_data'] = meta_data # store meta data per simulation

            if export:
                meta_data.to_csv(sim_dir + n_sample + '/' + run + '/meta_data.csv') # export meta_data to .csv
            
    return(data)

def create_dummies(data: pd.DataFrame):
    """
    Create dummy variables from categorical variables      
    """    
    
    for n_sample in data:
        
        for run in data[n_sample]:
            
            for dataset_name in [name for name in data[n_sample][run] if name not in ['meta_data']]:
                dataset = data[n_sample][run][dataset_name]       
                dataset['stage'] = pd.Categorical(dataset['stage'], categories=['I', 'II', 'III', 'IV'], ordered=True) # define all theoretical categories
                dataset_dummies = pd.get_dummies(dataset) # create dummies           
                data[n_sample][run][dataset_name] = dataset.merge(dataset_dummies) # merge dummies
    
    return(data)
            
def calculate_estimates(data: pd.DataFrame):
    """
    Calculate estimates (store in meta data per n_sample and per Monte Carlo run)     
    """  
    
    for n_sample in data:
        
        for run in data[n_sample]:
            
            # sample mean of age
            var = 'age'
            for estimator in ['mean', 'mean_se']:
                data[n_sample][run]['meta_data'][var + '_' + estimator] = estimate_estimator(
                    data=data[n_sample][run], var=var, estimator=estimator)
            
            # logistic regression coefficients of death in function of age, stage and therapy
            var = ('death', 'age', 'stage_II', 'stage_III', 'stage_IV', 'therapy') # death is outcome 
            tmp = np.array(estimate_estimator(data=data[n_sample][run], var=var, estimator='logr')).transpose() # calculate logistic regression coefficients
            columns = [var[0] + '_' + y + '_' + x for x, y in list(product(['logr', 'logr_se'], var[1:]))] # define names for regression coefficients
            for i in range(len(columns)):
                data[n_sample][run]['meta_data'][columns[i]] = list(tmp[i]) # assign corresponding names and values             
        
    return(data)

def combine_metadata(data: pd.DataFrame, export: bool=False):
    """
    Combine all meta data over n_sample and Monte Carlo runs    
    """    
    
    data['meta_data'] = pd.DataFrame({})
    for n_sample in data:
        if n_sample=='meta_data': # if overall meta_data was already created by previously calling combine_metadata() function
            continue
        for run in data[n_sample]:
            data['meta_data'] = pd.concat([data['meta_data'],
                                           data[n_sample][run]['meta_data']],
                                          ignore_index=True)

    if export:
        data['meta_data'].to_csv(sim_dir + 'meta_data.csv') # export meta_data to .csv
   
    return(data)

def estimates_check(meta_data: pd.DataFrame):
    """
    Check non-estimable estimates (too large or too small SE)
    """    
    
    select_se = [column for column in meta_data.columns if column[-3:]=='_se']  # select columns with '_se' suffix
    
    for var_se in select_se:
        
        # very small SE: set estimate and SE to np.nan
        meta_data[var_se[:-len('_se')]].mask(meta_data[var_se] < 1e-10, np.nan, inplace=True) 
        meta_data[var_se].mask(meta_data[var_se] < 1e-10, np.nan, inplace=True)
            
        # very large SE: set estimate and SE to np.nan
        meta_data[var_se[:-len('_se')]].mask(meta_data[var_se] > 1e2, np.nan, inplace=True)
        meta_data[var_se].mask(meta_data[var_se] > 1e2, np.nan, inplace=True)
        
    return(meta_data)

def inferential_utility(meta_data: pd.DataFrame, ground_truth):    
    """
    Calculate inferential utility metrics  
    """   
    
    # sample mean and logistic regression coefficients
    for estimator in ['age_mean', 
                      'death_age_logr', 'death_stage_II_logr', 'death_stage_III_logr', 'death_stage_IV_logr', 'death_therapy_logr']:
        
        # bias of population parameter
        meta_data[estimator + '_bias'] = meta_data.apply(
            lambda i:  i[estimator] - ground_truth[estimator], axis=1) # population parameter
                
        # rejection of population parameter (NHST type 1 error)
        meta_data[estimator + '_NHST_type1'] = meta_data.apply(
            lambda i:
            not CI_coverage(estimate=i[estimator],
                            se=i[estimator + '_se'],
                            ground_truth=ground_truth[estimator], # null hypothesis: mu = ground_truth
                            distribution='t' if 'mean' in estimator else 'standardnormal',
                            df=i['n']-1,
                            quantile=0.975), axis=1) # naive SE
        meta_data[estimator + '_NHST_type1_corrected'] = meta_data.apply(
            lambda i:
            not CI_coverage(estimate=i[estimator],
                            se=i[estimator + '_se'],
                            ground_truth=ground_truth[estimator], # null hypothesis: mu = ground_truth
                            se_correct_factor=np.sqrt(2) if i['dataset_name']!='original_data' else 1, # corrected SE
                            distribution='t' if 'mean' in estimator else 'standardnormal',
                            df=i['n']-1,
                            quantile=0.975), axis=1)    
        
    return(meta_data) 
      
if __name__ == "__main__":
    
    # Presets
    n_samples = [50, 160, 500, 1600, 5000] # number of observations per original data set
    n_runs = 200 # number of Monte Carlo runs per number of observations
    sim_dir = 'simulation_study1/' # output of simulations
    start_time = time()
    
    # Data preparation
    data = load_data(n_samples, n_runs, sim_dir) # load data
    data = create_metadata(data, export=False) # create (single) meta data per n_sample and Monte Carlo run
    data = create_dummies(data) # create dummy variables from categorical variables (needed for estimating regression coefficients)
    
    # Calculate estimates per data set
    data = calculate_estimates(data)
    
    # Combine (overall) meta data over n_sample and Monte Carlo runs 
    data = combine_metadata(data, export=False)
    
    # Only go further with meta_data (to reduce RAM memory)
    meta_data = data['meta_data']
    del data
    
    # Check non-estimable estimates (set to np.nan if too large or too small SE)
    meta_data = estimates_check(meta_data)

    # Calculate inferential utility metrics
    data_gt, _ = ground_truth()
    meta_data = inferential_utility(meta_data, ground_truth=data_gt)
    
    # Save to file
    meta_data.to_csv(sim_dir + 'meta_data.csv') # export meta_data to .csv
        
    # Print run time
    print(f'Total run time: {(time()-start_time):.3f} seconds')