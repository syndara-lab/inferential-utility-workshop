"""
Bayesian Network with DAG pre-specification integrated as synthcity module
Adapted from https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/plugins/generic/plugin_bayesian_network.py
"""

import pandas as pd
import numpy as np
from typing import Any, List

from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.core.dataloader import DataLoader

from synthcity.plugins.core.models.tabular_encoder import TabularEncoder 
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

class BayesianNetworkDAGPlugin(Plugin):
    """
    Bayesian Network with DAG prespecification for generative modeling. Implemented using pgmpy backend.
    Args:
        encoder_max_clusters: int = 10
            Data encoding clusters.
        encoder_noise_scale: float.
            Small noise to add to the final data, to prevent data leakage.
        dag: list.
            Pre-specified directed acyclic graph (DAG).
        compress_dataset: bool. Default = False.
            Drop redundant features before training the generator.
        random_state: int.
            Random seed.
        sampling_patience: int.
            Max inference iterations to wait for the generated data to match the training schema.
    """

    def __init__(
        self,
        encoder_max_clusters: int = 10,
        encoder_noise_scale: float = 0.1,
        dag: list = [], # NEW: add pre-specified DAG to module
        # core plugin
        compress_dataset: bool = False,
        random_state: int = 0,
        sampling_patience: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            random_state=random_state,
            sampling_patience=sampling_patience,
            compress_dataset=compress_dataset,
            **kwargs,
        )
        self.encoder = TabularEncoder(max_clusters=encoder_max_clusters)
        self.encoder_noise_scale = encoder_noise_scale
        self.dag = dag # NEW: add pre-specified DAG to module

    @staticmethod
    def name() -> str:
        return 'bayesian_network_DAG'

    @staticmethod
    def type() -> str:
        return 'custom'

    @staticmethod
    def hyperparameter_space(**kwargs: Any) -> List[Distribution]:
        return []

    def _encode_decode(self, data: pd.DataFrame) -> pd.DataFrame:
        encoded = self.encoder.transform(data)

        # add noise to the mixture means, but keep the continuous cluster
        noise = np.random.normal(
            loc=0, scale=self.encoder_noise_scale, size=len(encoded)
        )
        for col in encoded.columns:
            if col.endswith('.value'):
                encoded[col] += noise

        decoded = self.encoder.inverse_transform(encoded)
        decoded = decoded[data.columns]

        return decoded

    def _fit(self, X: DataLoader, *args: Any, **kwargs: Any) -> 'BayesianNetworkDAGPlugin':
        df = X.dataframe()
        self.encoder.fit(df)
        
        network = BayesianNetwork(self.dag)     
        network.fit(df) 

        self.model = BayesianModelSampling(network)
        return self

    def _generate(self, count: int, syn_schema: Schema, **kwargs: Any) -> pd.DataFrame:
        def _sample(count: int) -> pd.DataFrame:
            vals = self.model.forward_sample(size=count, show_progress=False)
            
            return self._encode_decode(vals)

        return self._safe_generate(_sample, count, syn_schema)