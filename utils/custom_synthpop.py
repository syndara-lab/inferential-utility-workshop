"""
R's synthpop integrated as synthcity module
Following https://github.com/vanderschaarlab/synthcity/blob/main/tutorials/tutorial1_add_a_new_plugin.ipynb
"""

from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.dataloader import DataLoader

import rpy2 # R in Python
import rpy2.robjects as ro 
import rpy2.robjects.packages 
import rpy2.robjects.pandas2ri 
base = ro.packages.importr('base') # import R's base package
synthpop = ro.packages.importr('synthpop') # import R's synthpop package

import pandas as pd
import numpy as np
from typing import Any, List

class synthpopPlugin(Plugin):
    """
    Sequential generative modeling. Implemented using R's synthpop backend.
    Args:
        random_state: int.
            Random seed.encoder_max_clusters: int = 10
        continuous_columns: list[str].
            Columns with continuous features.
        categorical_columns: list[str].
            Columns with categorical features.
        ordered_columns: list[str].
            Columns with ordered features.
        predictor_matrix: np.array.
            A square matrix specifying the set of column predictors to be used for each target variable in the row.
        default_method: list[str].
            A list of four strings containing the default parametric synthesising methods for numerical variables, factors with two levels, unordered factors with more than two levels 
            and ordered factors with more than two levels respectively.
    """
    
    def __init__(
        self,
        random_state: int = 0,
        continuous_columns: list = [str],
        categorical_columns: list = [str],
        ordered_columns: list = [str],
        predictor_matrix: np.array = np.array([0]),
        default_method: list = ['normrank', 'logreg', 'polyreg', 'polr'],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.random_state=random_state
        self.continuous_columns=continuous_columns
        self.categorical_columns=categorical_columns
        self.ordered_columns=ordered_columns
        self.predictor_matrix = predictor_matrix
        self.default_method = default_method

    @staticmethod
    def name() -> str:
        return 'synthpop'

    @staticmethod
    def type() -> str:
        return 'custom'

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> 'synthpopPlugin':
        
        ### data preparation
        
        # specify dtype
        vars_type = {}
        for i in self.categorical_columns + self.ordered_columns:
            vars_type[i] = 'object'
        for i in self.continuous_columns:
            vars_type[i] = 'float64'
        
        # convert pandas to R dataframe
        df = X.dataframe()
        with ro.conversion.localconverter(ro.default_converter + ro.pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df.astype(vars_type))
        r_colnames = base.colnames(r_df)
        
        # specify ordered and categorical columns
        if bool(set(self.ordered_columns) & set(r_colnames)):
            ordered_index = [int(np.where(np.array(base.colnames(r_df))==cat_name)[0][0]) for cat_name in self.ordered_columns]
            for i in ordered_index:
                r_df[i] = ro.FactorVector(r_df[i], ordered=True)
        
        if bool(set(self.categorical_columns) & set(r_colnames)):
            cat_index = [int(np.where(np.array(base.colnames(r_df))==cat_name)[0][0]) for cat_name in self.categorical_columns]
            for i in cat_index:
                r_df[i] = ro.FactorVector(r_df[i])
                
        # predictor matrix
        if len(self.predictor_matrix) == 1: # if predictor_matrix is not specificied
            predictor_matrix = np.tri(df.shape[1], k=-1).reshape(-1)
        else:
            predictor_matrix = self.predictor_matrix.reshape(-1)
        r_predictor_matrix = ro.r.matrix(ro.IntVector(predictor_matrix), ncol=df.shape[1], byrow=True, dimnames=[list(df.columns), list(df.columns)])
        
        # default synthesising methods
        r_default_method = ro.StrVector(self.default_method)      
        
        # arguments for synthpop.syn (fitting is done simultaneously with generation below)
        self.params = {'data': r_df,
                       'method': 'parametric',
                       'predictor.matrix': r_predictor_matrix,
                       'proper': False,
                       'm': 1,
                       'smoothing': 'spline',
                       'default.method': r_default_method,
                       'print.flag': False}
               
        return self
    
    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> DataLoader:
        
        self.params['seed'] = kwargs['seed'] if 'seed' in kwargs else self.random_state # self.random_state cannot be overwritten in _generate(), so use 'seed' argument for synthpopPlugin when generating multiple synthetic datasets
        self.params['seed'] = ro.IntVector([self.params['seed']]) # use different seed per generated dataset (when setting m=1), otherwise synthpop introduces a small bias (also in R)!
        self.model = synthpop.syn(k=count, **self.params)
                
        def _sample(count: int) -> pd.DataFrame:
            with ro.conversion.localconverter(ro.default_converter + ro.pandas2ri.converter):
                r_synthetic_df = self.model.rx2('syn')
                X_rnd = ro.conversion.rpy2py(r_synthetic_df)
            return X_rnd

        return self._safe_generate(_sample, count, syn_schema)