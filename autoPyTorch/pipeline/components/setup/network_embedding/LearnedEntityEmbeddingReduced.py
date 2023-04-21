from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)

import numpy as np

import torch
from torch import embedding, nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_embedding.base_network_embedding import NetworkEmbeddingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter
from autoPyTorch.pipeline.components.setup.network_embedding.utils import _LearnedEntityEmbedding


class LearnedEntityEmbeddingReduced(NetworkEmbeddingComponent):
    """
    Class to learn an embedding for categorical hyperparameters.
    """

    def __init__(self, random_state: Optional[np.random.RandomState] = None, **kwargs: Any):
        super().__init__(random_state=random_state)
        self.config = kwargs

       def build_embedding(self, num_categories_per_col: np.ndarray, num_features_excl_embed: int) -> Tuple[_LearnedEntityEmbedding, int]:

        embedding = _LearnedEntityEmbedding(config=self.config,
                                       num_categories_per_col=num_categories_per_col,
                                       num_features_excl_embed=num_features_excl_embed,
                                       reduced=True)

        return embedding, embedding.num_output_dimensions

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        embed_exponent: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="embed_exponent",
                                                                                   value_range=(0.56,),
                                                                                   default_value=0.56),
        max_embedding_dim: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="max_embedding_dim",
                                                                                   value_range=(100,),
                                                                                   default_value=100),
        embedding_size_factor: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="embedding_size_factor",
                                                                                     value_range=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
                                                                                     default_value=1,
                                                                                     ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        if dataset_properties is not None:
            if len(dataset_properties['categorical_columns']) > 0:
                add_hyperparameter(cs, embed_exponent, UniformFloatHyperparameter)
                add_hyperparameter(cs, max_embedding_dim, UniformIntegerHyperparameter)
                add_hyperparameter(cs, embedding_size_factor, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'embedding',
            'name': 'LearnedEntityEmbedding',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': True,
        }
