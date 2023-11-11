# Synthetic Data: Can We Trust Statistical Estimators?
Code to reproduce results in "Synthetic Data: Can We Trust Statistical Estimators?", presented during the 1st Workshop on Deep Generative Models for Health at NeurIPS 2023.

In this work, we highlight the importance of inferential utility and provide empirical evidence against naive inference from synthetic data (that handles these as if they were really observed). We argue that the rate of false-positive findings (type 1 error) will be unacceptably high, even when the estimates are unbiased. One of the reasons is the underestimation of the true standard error, which may even progressively increase with larger sample sizes due to slower convergence. This is especially problematic for deep generative models. Before publishing synthetic data, it is essential to develop statistical inference tools for such data.

## Experiments
The following class and helper files are included: 
- hpo_results/: sql-lite databases containing the hyperparameter optimization studies for CTGAN and TVAE (using optuna backend) 
- utils/custom_bayesian.py: class to train Bayesian Network with DAG pre-specification (using synthcity and pgpmy backend)
- utils/custom_ctgan.py: class to train CTGAN (using sdv backend)
- utils/custom_synthpop.py: class to train Synthpop (using synthcity and R's synthpop backend)
- utils/custom_tvae.py: class to train TVAE (using sdv backend)
- utils/disease.py: simulate low-dimensional tabular toy data sampled from an arbitrary ground truth population
- utils/eval.py: functions to calculate and plot inferential utility metrics

Run experiments: 
- sim_generate.py: sample original data and generate synthetic version(s) using different generative models
- sim_evaluate.py: calculate inferential utility metrics of original and synthetic datasets
- sim_output.ipynb: notebook containing all output (figures and tables) presented in paper

## Cite
If our paper or code helped you in your own research, please cite our work as:

```
@inproceedings{decruyenaere2023synthetic,
  title={Synthetic Data: Can We Trust Statistical Estimators?},
  author={Decruyenaere, Alexander and Dehaene, Heidelinde and Rabaey, Paloma and Polet, Christiaan and Decruyenaere, Johan and Vansteelandt, Stijn and Demeester, Thomas},
  year={2023},
  booktitle={1st Workshop on Deep Generative Models for Health},
  organization={NeurIPS}
}
```
