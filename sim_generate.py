"""Simulation study 1: sample original data and generate synthetic version(s) using different generative models"""

import numpy as np
import os
from time import time
from utils.disease import sample_syndara_disease
from utils.custom_ctgan import CTGAN
from utils.custom_tvae import TVAE
from utils.custom_bayesian import BayesianNetworkDAGPlugin
from utils.custom_synthpop import synthpopPlugin
from optuna import load_study
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

def generative_models(n_sample: int=200):
    """
    Specify generative models used in simulation study
    """    
    
    # Extract tuned hyperparameters for CTGAN
    study_ctgan = load_study(study_name='CTGAN_custom_sdv_study_multiple', storage=f'sqlite:///hpo_results/CTGAN_custom_sdv_study_multiple.db') # load study
    study_ctgan_df = study_ctgan.trials_dataframe()
    hyperparam_keys = [column for column in study_ctgan_df.columns if 'params_' in column] # select hyperparameter columns
    hyperparams_ctgan = study_ctgan_df.sort_values(by=['value'], ascending=False).loc[2, hyperparam_keys].to_dict() # select hyperparameters of third best trial
    hyperparams_ctgan = {k[len('params_'):]: hyperparams_ctgan[k] for k in hyperparams_ctgan.keys()} # remove 'params_' prefix    
    hyperparams_ctgan['discriminator_lr'] = hyperparams_ctgan['generator_lr'] # set discriminator_lr equal to generator_lr
    hyperparams_ctgan['generator_dim'] = hyperparams_ctgan['generator_n_layers_hidden']*(hyperparams_ctgan['generator_n_units_hidden'],) # define generator layer dimensions
    hyperparams_ctgan['discriminator_dim'] = hyperparams_ctgan['discriminator_n_layers_hidden']*(hyperparams_ctgan['discriminator_n_units_hidden'],) # define discriminator layer dimensions
    hyperparams_ctgan['batch_size'] = min(n_sample, 200)
    for name in ['discriminator_n_layers_hidden', 'discriminator_n_units_hidden', 'generator_n_layers_hidden', 'generator_n_units_hidden']: # use the correct naming of hyperparameters in the CTGAN module
        del hyperparams_ctgan[name] 

    # Extract tuned hyperparameters for TVAE
    study_tvae = load_study(study_name='TVAE_custom_sdv_study_multiple', storage=f'sqlite:///hpo_results/TVAE_custom_sdv_study_multiple.db') # load study
    hyperparams_tvae = study_tvae.best_params # select hyperparameters of best trial
    hyperparams_tvae['compress_dims'] = hyperparams_tvae['n_layers']*(hyperparams_tvae['n_hidden_units'],)  # define encoder layer dimensions
    hyperparams_tvae['decompress_dims'] = hyperparams_tvae['n_layers']*(hyperparams_tvae['n_hidden_units'],)  # define decoder layer dimensions (here equal to encoder)
    hyperparams_tvae['batch_size'] = min(n_sample, 200)
    for name in ['n_layers', 'n_hidden_units']: # use the correct naming of hyperparameters in the TVAE module
        del hyperparams_tvae[name]
    
    # Define generative models (assign synthcity Plugins() to synthcity_models object prior to calling this function)
    models = [synthcity_models.get('synthpop',
                                   continuous_columns=['age', 'biomarker'],
                                   categorical_columns=['therapy', 'death'],
                                   ordered_columns=['stage'],
                                   predictor_matrix=np.array([[0,0,0,0,0],
                                                              [1,0,0,0,0],
                                                              [0,1,0,0,0],
                                                              [0,0,0,0,0],
                                                              [1,1,0,1,0]])), # age stage biomarker therapy death
              synthcity_models.get('bayesian_network_DAG',
                                   dag=[('age', 'stage'), ('stage', 'biomarker'), ('age', 'death'), ('stage', 'death'), ('therapy', 'death')]),        
              CTGAN(**hyperparams_ctgan), # tuned hyperparameters
              TVAE(**hyperparams_tvae)] # tuned hyperparameters 
  
    return models

def simulation_study1(n_samples, n_runs, n_sets, sim_dir, discrete_columns):
    """
    Setup of simulation study
    """    
    
    # OUTER loop over number of observations per original data set
    for n_sample in n_samples: 
        
        # Define output folder to save files
        n_dir = f'n_{n_sample}/'
        if not os.path.exists(sim_dir + n_dir): 
            os.mkdir(sim_dir + n_dir) # create n_dir folder if it does not exist yet
            
        # Define generative models
        models = generative_models(n_sample) # batch_size hyperparameter depends on sample size

        # INNER loop over Monte Carlo runs        
        for i in range(n_runs):
            
            # Print progress
            print(f'[n={n_sample}, run={i}] start')
            
            # Define output folder to save files
            out_dir = sim_dir + n_dir + f'run_{i}/'
            if not os.path.exists(out_dir): 
                os.mkdir(out_dir) # create out_dir folder if it does not exist yet
            
            # Simulate toy data
            original_data = sample_syndara_disease(n_sample, seed=i)
            original_data.to_csv(out_dir + 'original_data.csv', index=False) # export file           
            
            # Train generative models
            for model in models:
                
                try:
                    
                    # Use custom plugins (adapted from SDV)
                    if 'custom' in model.name():
                        model.fit(original_data, discrete_columns=discrete_columns) # seed is fixed internally

                    # Use synthcity plugins
                    else:
                        loader = GenericDataLoader(original_data, sensitive_features=list(original_data.columns))
                        model.fit(loader) # seed is fixed internally
                
                except Exception as e:
                    print(f'[n={n_sample}, run={i}] error with fitting {model.name()}: {e}')
            
            # Generate synthetic data
            for model in models:
                
                try:
                
                    # Use custom plugins (adapted from SDV)
                    if 'custom' in model.name(): 
                        for j in range(n_sets):
                            synthetic_data = model.sample(n=n_sample, seed=j) # generated synthetic data = size of original data                         
                            synthetic_data.to_csv(out_dir + model.name() + f'_{j}.csv', index=False) # export file           

                    # Use synthcity plugins
                    else: 
                        for j in range(n_sets):
                            synthetic_data = model.generate(count=n_sample, # generated synthetic data = size of original data
                                                            seed=i, # seed=i argument is used by synthpopPlugin
                                                            random_state=j).dataframe() # random_state=j is used by other plugins
                            synthetic_data.to_csv(out_dir + model.name() + f'_{j}.csv', index=False) # export file
                
                except Exception as e:
                    print(f'[n={n_sample}, run={i}] error with generating from {model.name()}: {e}')
            
if __name__ == "__main__":
    
    # Presets 
    n_samples = [50, 160, 500, 1600, 5000] # number of observations per original dataset
    n_runs = 200 # number of Monte Carlo runs per number of observations
    n_sets = 1 # number of synthetic datasets generated per generative model
    
    sim_dir = 'simulation_study1/' # output of simulations
    if not os.path.exists(sim_dir): 
        os.mkdir(sim_dir) # create sim_dir folder if it does not exist yet
    
    discrete_columns = ['stage', 'therapy', 'death'] # columns with discrete values
    
    # Disable tqdm as default setting (used internally in synthcity plugins)
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    # Load synthcity plugins 
    synthcity_models = Plugins()
    synthcity_models.add('bayesian_network_DAG', BayesianNetworkDAGPlugin) # add the BayesianNetworkDAGPlugin to the collection
    synthcity_models.add('synthpop', synthpopPlugin) # add the synthpopPlugin to the collection

    # Run simulation study
    start_time = time()
    simulation_study1(n_samples, n_runs, n_sets, sim_dir, discrete_columns)
    print(f'Total run time: {(time()-start_time):.3f} seconds')