import pandas as pd
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.api import OLS
import itertools
from itertools import product
import plotnine
from plotnine import *
from mizani.formatters import percent_format 
from mizani.transforms import trans

def estimate_estimator(data, var, estimator):
    """
    Calculate estimate for each dataset
    """    
    
    datasets = [data[name] for name in data if name not in ['meta_data']]

    if estimator == 'mean':
        return [np.mean(dataset[var]) for dataset in datasets]
    elif estimator == 'mean_se':
        return [np.std(dataset[var], ddof=1)/np.sqrt(len(dataset[var])) for dataset in datasets]

    elif estimator == 'logr':
        output = []
        for dataset in datasets:
            try:
                model = GLM(dataset[var[0]], sm.add_constant(dataset.loc[:,var[1:]].astype(float)), family=sm.families.Binomial(link=sm.families.links.Logit())).fit() # var[0] is outcome, var[1:] are predictors
                indices = model.cov_params().index.get_indexer(var[1:])
                output.append(np.hstack((model.params.to_numpy()[indices], # coefficients
                                         np.sqrt(model.cov_params().to_numpy().diagonal()[indices])))) # SEs
            except Exception as e:
                output.append(np.repeat(np.nan, len(var[1:])*2))
        return output

    else:
        return [np.nan for dataset in datasets]
    
def CI_coverage(estimate, se, ground_truth, se_correct_factor=1, distribution='standardnormal', df=1, quantile=0.975):
    """
    Calculate coverage of a (1-quantile)*100% confidence interval based on t- or standard normal distribution
    """  
    
    # Define quantile based on distribution
    if distribution == 't':
        q = ss.t.ppf(quantile, df=df)  
    elif distribution == 'standardnormal': 
        q = ss.norm.ppf(quantile)
    else:
        raise ValueError('Choose \'t\' or \'standardnormal\' distribution')

    # Check if confidence interval contains ground_truth
    if (estimate-q*se_correct_factor*se <= ground_truth) & (ground_truth <= estimate+q*se_correct_factor*se):
        coverage = True
    else:
        coverage = False
    
    return(coverage)

def q1(x):
    """
    First quartile
    """ 
    return x.quantile(0.25)

def q3(x):
    """
    Third quartile
    """ 
    return x.quantile(0.75)

def expand_grid(data_dict):
    """
    Expand grid
    """ 
    rows = product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def plot_bias(meta_data: pd.DataFrame, select_estimators: list=[], plot_outliers: bool=True, order_generators: list=[], figure_size: tuple=(), 
              unit_rescale: dict={}, plot_estimates: bool=False, ground_truth: dict={}):
    """
    Plot bias. Note that pd.DataFrame.groupby.mean ignores missing values when calculating the mean (desirable).
    If plot_estimates==True, then ground_truth should be given.
    """ 
    
    # Select estimators
    suffix = '_bias'
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if suffix in column] # select all columns with '_bias' suffix
    else:
        select_estimators = [estimator + suffix for estimator in select_estimators] # add suffix '_bias'
    
    # Plot estimates instead of bias
    if plot_estimates:
        select_estimators = [estimator[:-len('_bias')] for estimator in select_estimators] # remove suffix '_bias'
        suffix = ''
        
    # Average bias/estimate of estimator for population parameter per generator (over sets) for each n and run
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias')

    # Don't plot outliers based on IQR of empirical SE (set plot_outliers = False)
    if plot_outliers:
        bias_data['plot'] = True
    else:
        iqr = meta_data.groupby(['n', 'generator'])[select_estimators].agg([q1, q3]).reset_index()
        iqr.columns = ['_'.join(column) if column[1] != '' else column[0] for column in iqr.columns.to_flat_index()] # rename hierarchical columns
        iqr = iqr.melt(
            id_vars=['n', 'generator'],
            var_name='estimator',
            value_name='value')
        iqr['quantile'] = iqr.apply(lambda x: 'q1' if 'q1' in x['estimator'] else 'q3', axis=1)
        iqr['estimator'] = list(map(lambda x: x[:-3], iqr['estimator']))
        iqr = iqr.pivot(index=['n', 'generator', 'estimator'], columns='quantile', values='value').reset_index()
        bias_data = bias_data.merge(iqr, how='left', on=['n', 'generator', 'estimator'])
        bias_data['plot'] = bias_data.apply(
            lambda x: x['bias'] > x['q1'] - 1.5*(x['q3']-x['q1']) and x['bias'] < x['q3'] + 1.5*(x['q3']-x['q1']), axis=1) # non-outlier is > Q1-1.5*IQR and < Q3+1.5*IQR
        

    # Change plotting order (non-alphabetically) of estimator and generator
    bias_data['estimator'] = pd.Categorical(list(map(lambda x: x[:(-len(suffix) or None)], bias_data['estimator'])), # remove suffix; 'or None' is called when suffix=''
                                            categories=[estimator[:(-len(suffix) or None)] for estimator in list(bias_data['estimator'].unique())]) # change order (non-alphabetically); 'or None' is called when suffix=''
    if len(order_generators)!=0:
        bias_data['generator'] = pd.Categorical(bias_data['generator'], 
                                                categories=order_generators) # change order (non-alphabetically)

    # Root-n consistency funnel
    root_n_consistency = expand_grid({'x': np.arange(np.min(bias_data['n']), np.max(bias_data['n'])), 'estimator': bias_data['estimator'].unique()})
    root_n_consistency['estimator'] = pd.Categorical(root_n_consistency['estimator'], categories=root_n_consistency['estimator'].unique()) # change order (non-alphabetically)
    root_n_consistency['unit_sd'] = root_n_consistency.apply(lambda x: unit_rescale[x['estimator']], axis=1) # asymptotic variance
    root_n_consistency['y'] =  root_n_consistency.apply(lambda x: ground_truth[x['estimator']], axis=1) if plot_estimates else 0 
    root_n_consistency['y_ul'] = root_n_consistency['y'] + ss.norm.ppf(0.975)*root_n_consistency['unit_sd']/np.sqrt(root_n_consistency['x'])
    root_n_consistency['y_ll'] = root_n_consistency['y'] - ss.norm.ppf(0.975)*root_n_consistency['unit_sd']/np.sqrt(root_n_consistency['x'])
    
    # Labs
    plot_title = 'Estimates of estimator' if plot_estimates else 'Bias of estimator'
    plot_y_lab = 'Estimate' if plot_estimates else 'Bias'
    
    # Default figure size
    if len(figure_size) == 0:
        figure_size = (1.5+len(bias_data['generator'].unique())*1.625, 1.5+len(bias_data['estimator'].unique())*1.625)
    
    # Plot average bias/estimate and root-n consistency funnel
    plot = ggplot(bias_data.query('plot==True'), aes(x='n', y='bias', colour='generator')) +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y'), linetype='dashed', colour='black') +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y_ul'), linetype='dashed', colour='black') +\
        geom_line(data=root_n_consistency, mapping=aes(x='x', y='y_ll'), linetype='dashed', colour='black') +\
        geom_point(alpha=0.20) +\
        stat_summary(geom='line') +\
        scale_x_continuous(breaks=list(bias_data['n'].unique()), labels=list(bias_data['n'].unique()), trans='log') +\
        facet_grid('estimator ~ generator', scales='free') +\
        scale_colour_manual(values={'original': '#808080', 'synthpop': '#1E64C8', 'bayesian_network_DAG': '#71A860', 'custom_ctgan': '#F1A42B', 'custom_tvae': '#FFD200'}) +\
        labs(title=plot_title, x='n (log scale)', y=plot_y_lab) +\
        theme_bw() +\
        theme(plot_title=element_text(hjust=0.5, size=12), # title size
              axis_title=element_text(size=10), # axis title size
              strip_text=element_text(size=8), # facet_grid title size
              axis_text=element_text(size=8), # axis labels size
              legend_position='none',
              figure_size=figure_size)
    
    return plot

class log_squared(trans):
    """
    Custom axis transformation: transform x to log(x**2)
    """
    @staticmethod
    def transform(x):
        return np.log(x**2)
    
    @staticmethod
    def inverse(x):
        return np.sqrt(np.exp(x))
    
def plot_convergence_rate(meta_data: pd.DataFrame, select_estimators: list=[], figure_ncol: int=None, order_generators: list=[], figure_size: tuple=(), 
                          unit_rescale: dict={}, metric: str='se', check_root_n: bool=True):
    """
    Plot convergence rate
    """ 
    
    # Select estimators
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if '_bias' in column] # select all columns with '_bias' suffix
    else:
        select_estimators = [estimator + '_bias' for estimator in select_estimators] # add suffix '_bias'
                          
    # Empirical SE/bias of estimator for population parameter per generator (over sets) for each n and run
    if metric == 'se':
        plot_data = meta_data.groupby(['n', 'generator'])[select_estimators].apply(lambda x: np.std(x, ddof=1)).reset_index().melt(
            id_vars=['n', 'generator'],
            var_name='estimator',
            value_name='metric')
        if check_root_n:
            plot_y_axis = 'se_rescaled*np.sqrt(n)'
            plot_y_lab = 'empirical SE * sqrt(n)'
            plot_x_axis = 'n' # the x-axis will be transformed to log-scale in the plot
            plot_x_lab = 'n (log scale)'
            plot_x_trans = 'log'
        else:
            plot_y_axis = 'np.log(metric)'
            plot_y_lab = 'log(empirical SE)'
            plot_x_axis = 'n' # the x-axis will be transformed to log-scale in the plot
            plot_x_lab = 'n (log scale)'
            plot_x_trans = 'log'
            
    elif metric == 'bias':
        plot_data = meta_data.groupby(['n', 'generator'])[select_estimators].mean().reset_index().melt(
            id_vars = ['n', 'generator'],
            var_name = 'estimator',
            value_name = 'metric')
        if check_root_n:
            plot_y_axis = 'se_rescaled*n'
            plot_y_lab = 'empirical SE * n'
            plot_x_axis = 'n' # the x-axis will be transformed to log-scale in the plot
            plot_x_lab = 'n (log scale)'
            plot_x_trans = 'log'
        else:
            plot_y_axis = 'np.log(metric**2)'
            plot_y_lab = 'log(empirical SE**2)'
            plot_x_axis = 'n**2' # the x-axis will be transformed to log-scale in the plot
            plot_x_lab = 'n**2 (log scale)'
            plot_x_trans = log_squared

    # Rescale estimate with asymptotic variance
    plot_data['unit_sd'] = plot_data.apply(lambda x: unit_rescale[x['estimator'][:-len('_bias')]], axis=1)
    plot_data['se_rescaled'] = plot_data['metric']/plot_data['unit_sd']
    
    # Custom breaks and labels
    plot_x_breaks = list(plot_data['n'].unique()**2) if (metric=='bias' and  not check_root_n) else list(plot_data['n'].unique())
    plot_x_labels = list(plot_data['n'].unique())
    
    # Change plotting order (non-alphabetically) of estimator and generator
    plot_data['estimator'] = pd.Categorical(plot_data['estimator'], categories=list(plot_data['estimator'].unique())) # change order (non-alphabetically)
    if len(order_generators)!=0:
        plot_data['generator'] = pd.Categorical(plot_data['generator'], categories=order_generators) # change order (non-alphabetically)
    
    # Default figure columns and size
    if figure_ncol == None:
        figure_ncol = 7
        
    if len(figure_size) == 0:
        figure_size = (16,8)

    # Plot
    plot = ggplot(plot_data, aes(x=plot_x_axis, y=plot_y_axis, colour='generator'))

    # Plot root-n-convergence of naive and corrected SE (if metric == 'se' and not check_root_n)
    if metric == 'se' and not check_root_n:
        
        # Create additional dataframe with naive and corrected SE
        constant_c = plot_data.drop_duplicates(subset=['estimator', 'unit_sd'])[['estimator', 'unit_sd']]
        constant_c['formula_se'] = 'naive'       
        constant_c_corrected = constant_c.copy()
        constant_c_corrected['formula_se'] = 'corrected'
        constant_c_corrected['unit_sd'] *= np.sqrt(2)
        constant_c = pd.concat([constant_c, constant_c_corrected], axis=0)
        constant_c['formula_se'] = pd.Categorical(constant_c['formula_se'], categories=list(constant_c['formula_se'].unique()))
        
        # Add abline to plot
        plot += geom_abline(data=constant_c, mapping=aes(intercept='np.log(unit_sd)', slope=-0.5, linetype='formula_se'), colour='black') 
    
    # Continue plot building
    plot = plot +\
        geom_point() +\
        stat_smooth(method='lm', se=False) +\
        scale_x_continuous(breaks=plot_x_breaks, labels=plot_x_labels, trans=plot_x_trans) +\
        facet_wrap('estimator', ncol=figure_ncol, scales='free')  +\
        scale_colour_manual(values={'original': '#808080', 'synthpop': '#1E64C8', 'bayesian_network_DAG': '#71A860', 'custom_ctgan': '#F1A42B', 'custom_tvae': '#FFD200'}) +\
        labs(title='Convergence of estimator', x=plot_x_lab, y=plot_y_lab) +\
        theme_bw() +\
        theme(plot_title=element_text(hjust=0.5, size=12), # title size
              axis_title=element_text(size=10), # axis title size
              strip_text=element_text(size=8), # facet_grid title size
              axis_text=element_text(size=8), # axis labels size
              figure_size=figure_size)
    
    # Add legend for root-n-convergence of naive and corrected SE (if metric == 'se' and not check_root_n)
    if metric == 'se' and not check_root_n:
        plot += scale_linetype_manual(values={'naive': 'dashed', 'corrected': 'dotted'})

    return plot

def table_convergence_rate(meta_data: pd.DataFrame, select_estimators: list=[], unit_rescale: dict={}, metric: str='se', round_decimals: int=2, 
                           show_ci: bool=False, quantile: int=0.975):
    """
    Table with convergence rates
    """ 
    
    # Calculate consistency convergence rate d: if d=1/2 then root-n-consistency; decreasing d -> slower convergence rate
    def calculate_conv_rate(data, generator, estimator, metric, round_decimals, show_ci, quantile):
        
        # SE
        if metric == 'se':
            empirical_se = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].apply(lambda x: np.std(x, ddof=1)).reset_index()
            unit_sd = unit_rescale[estimator[:-len('_bias')]] # scale estimate to measurument unit - predefine outside function
            log_lr = OLS(np.log(empirical_se['bias']/unit_sd), np.log(empirical_se['n'])).fit() # an intercept is not included by default         
        
        # bias
        elif metric == 'bias':
            empirical_bias = data.query(f'generator==\'{generator}\' & estimator==\'{estimator}\'').groupby('n')['bias'].mean().reset_index()
            unit_sd = unit_rescale[estimator[:-len('_bias')]] # rescale estimate with asymptotic variance (prespecify outside function)
            log_lr = OLS(np.log((empirical_bias['bias']/unit_sd)**2), np.log(empirical_bias['n']**2)).fit() # an intercept is not included by default
        
        # extract rate and rate_se
        rate = -log_lr.params[0]
        rate_se = np.sqrt(log_lr.cov_params().to_numpy().diagonal()[0])
        rate_q = ss.t.ppf(quantile, df=(log_lr.nobs-1))
        rate_ll = rate-rate_q*rate_se
        rate_ul = rate+rate_q*rate_se
        
        # round decimals
        rate = f'{rate:.{round_decimals}f}'
        rate_ll = f'{rate_ll:.{round_decimals}f}'
        rate_ul = f'{rate_ul:.{round_decimals}f}'
        
        output = f'{rate} [{rate_ll}; {rate_ul}]' if show_ci else rate
        
        return output

    # Select estimators
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if '_bias' in column] # select all columns with '_bias' suffix
    else:
        select_estimators = [estimator + '_bias' for estimator in select_estimators] # add suffix '_bias'

    # Average bias of estimator for population parameter per generator (over sets) for each n and run
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias')

    # Calculate convergence rate for every estimator
    output_data = expand_grid({'generator': bias_data['generator'].unique(),
                               'estimator': bias_data['estimator'].unique()})
    output_data['convergence rate'] = output_data.apply(
        lambda x: calculate_conv_rate(data=bias_data, generator=x['generator'], estimator=x['estimator'], metric=metric, round_decimals=round_decimals, 
                                      show_ci=show_ci, quantile=quantile), axis=1)
    
    return output_data

def se_underestimation(meta_data: pd.DataFrame, select_estimators: list=[], correction_factor: int=1):
    """
    Calculate underestimation of empirical SE by model-based SE.
    """

    # Select estimators
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if 'bias' in column] # select all columns with '_bias' suffix
    else:
        select_estimators = [estimator + '_bias' for estimator in select_estimators] # add suffix '_bias'
        
    # Average bias of estimator for population parameter per generator (over sets) for each n and run
    bias_data = meta_data.groupby(['n', 'run', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'run', 'generator'],
        var_name='estimator',
        value_name='bias') 
    
    # Create dataset
    output_data = expand_grid({'generator': bias_data['generator'].unique(),
                               'estimator': bias_data['estimator'].unique()})    
    
    # Empirical SE
    empirical_se = output_data.apply(lambda x: 
                                     bias_data.query('generator==\'' + x['generator'] + '\' & estimator==\'' + x['estimator'] + '\'').groupby('n')['bias'].apply(lambda x: np.std(x, ddof=1)),
                                     axis=1) 
    
    # Model-based SE  
    model_based_se = output_data.apply(lambda x: 
                                       meta_data.query('generator==\'' + x['generator'] + '\'').groupby('n')[x['estimator'].replace('_bias', '_se')].mean() * correction_factor,
                                       axis=1) 
      
    # Metric 
    metric = (model_based_se-empirical_se)/empirical_se
    
    # Add metric to dataset
    output_data = pd.concat([output_data, metric], axis=1)
    
    return output_data

def summary_table(meta_data: pd.DataFrame, select_estimators: list=[], ground_truth: dict={}, correction_factor: int=1):
    """
    Create summary table
    """ 

    # Select estimators
    if len(select_estimators)==0:
        select_estimators = [column for column in meta_data.columns if 'bias' in column] # select all columns with '_bias' suffix
        select_estimators_no_suffix = [name[:-len('_bias')] for name in select_estimators] # remove '_bias' suffix
    else:
        select_estimators_no_suffix = select_estimators
        select_estimators = [estimator + '_bias' for estimator in select_estimators] # add suffix '_bias'
        
    # Average estimate of population parameter per generator (over sets and runs) for each n
    estimate_data = meta_data.groupby(['n', 'generator'])[select_estimators_no_suffix].mean().reset_index().melt(
        id_vars=['n', 'generator'],
        var_name='estimator',
        value_name='average')

    # Average absolute bias of estimator for population parameter per generator (over sets and runs) for each n
    bias_data = meta_data.groupby(['n', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'generator'],
        var_name='estimator',
        value_name='absolute_bias')
    bias_data['estimator'] = list(map(lambda x: x[:-len('_bias')], bias_data['estimator'])) # remove '_bias' suffix

    # Average relative bias
    bias_data['relative_bias'] = bias_data.apply(lambda x: x['absolute_bias']/ground_truth[x['estimator']], axis=1) 

    # Merge estimate_data and bias_data
    output_data = estimate_data.merge(bias_data, how='left', on=['n', 'generator', 'estimator'])

    # SE underestimation
    se_underest = se_underestimation(meta_data, select_estimators_no_suffix, correction_factor=correction_factor).melt(
        id_vars=['estimator', 'generator'],
        var_name='n',
        value_name='SE_underestimation')
    se_underest['estimator'] = list(map(lambda x: x[:-len('_bias')], se_underest['estimator'])) # remove '_bias' suffix

    # Merge output_data and se_underest
    output_data = output_data.merge(se_underest, how='left', on=['n', 'generator', 'estimator'])

    return output_data

def plot_type_I_II_error(meta_data: pd.DataFrame, select_estimator: str, order_generators: list=[], use_power: bool=False, plot_intercept: bool=True, figure_size: tuple=()):
    """
    Plot type I and II error rate
    """     
    
    # Select estimators
    select_estimators = [column for column in meta_data.columns if select_estimator + '_NHST' in column] # add suffix '_NHST'
    
    # Average type 1 and type 2 error rate per generator (over sets and runs) for each n
    NHST_data_plot = meta_data.groupby(['n', 'generator'])[select_estimators].mean().reset_index().melt(
        id_vars=['n', 'generator'],
        var_name='error',
        value_name='probability')   

    # Change plotting order (non-alphabetically)
    NHST_data_plot['corrected'] = pd.Categorical(list(map(lambda x: 'corrected SE' if 'corrected' in x else 'model-based SE', NHST_data_plot['error'])), # change label
                                                 categories=['model-based SE', 'corrected SE']) # change order (non-alphabetically)
    if len(order_generators)!=0:
        NHST_data_plot['generator'] = pd.Categorical(NHST_data_plot['generator'],
                                                     categories=order_generators) # change order (non-alphabetically)    
       
    # Additional plot formatting 
    NHST_data_plot['error'] = pd.Categorical(list(map(lambda x: 'type 1 error' if 'type1' in x else ('power' if use_power else 'type 2 error'), NHST_data_plot['error'])), # change label
                                             categories=['type 1 error', 'power' if use_power else 'type 2 error']) # change order (non-alphabetically) 
    NHST_data_plot['probability'] = NHST_data_plot.apply(lambda x:  1-x['probability'] if x['error']=='power' else x['probability'], axis=1) # power = 1 - type 2 error
    
    
    # Default figure size
    if len(figure_size) == 0:
        figure_size = (8,6)
        
    # Plot
    plot = ggplot(NHST_data_plot, aes(x='n', y='probability', colour='generator', linetype='generator'))
    
    # Plot intercepts (if plot_intercept)
    if plot_intercept:
        intercepts = pd.DataFrame({'error': ['type 1 error', 'power' if use_power else 'type 2 error'], 'intercept': [0.05, 0.80 if use_power else 0.20]}) # add horizontal lines
        intercepts['error'] = pd.Categorical(intercepts['error'], categories=['type 1 error', 'power' if use_power else 'type 2 error']) # change order (non-alphabetically) 
        plot += geom_hline(data=intercepts, mapping=aes(yintercept='intercept'), linetype='dashed')
        
    # Continue plot building
    plot = plot +\
        geom_line() +\
        scale_x_continuous(breaks=list(NHST_data_plot['n'].unique()), labels=list(NHST_data_plot['n'].unique()), trans='log') +\
        scale_y_continuous(limits=(0,1), labels=percent_format()) +\
        facet_grid('error ~ corrected')  +\
        scale_colour_manual(values={'original': '#808080', 'synthpop': '#1E64C8', 'bayesian_network_DAG': '#71A860', 'custom_ctgan': '#F1A42B', 'custom_tvae': '#FFD200'}) +\
        scale_linetype_manual(values={'original': 'solid', 'synthpop': 'solid', 'bayesian_network_DAG': 'solid', 'custom_ctgan': 'dashed', 'custom_tvae': 'dashed'}) +\
        labs(title='Type 1 error and ' + ('power' if use_power else 'type 2 error') + ' for ' + select_estimator, x='n (log scale)') +\
        theme_bw() +\
        theme(plot_title=element_text(hjust=0.5, size=12), # title size
              axis_title=element_text(size=10), # axis title size
              strip_text=element_text(size=8), # facet_grid title size
              axis_text=element_text(size=8), # axis labels size
              figure_size=figure_size) 
   
    return plot