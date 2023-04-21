from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)

import numpy as np


from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_embedding.base_network_embedding import NetworkEmbeddingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter
from autoPyTorch.pipeline.components.setup.network_embedding.utils import _LearnedEntityEmbedding

class LearnedEntityEmbedding(NetworkEmbeddingComponent):
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
                                       reduced=False)
        self.num_out_feats = embedding.num_out_feats
        return embedding, embedding.num_output_dimensions

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, Any]] = None,
        dimension_reduction: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="dimension_reduction",
            value_range=(0, 1),
            default_value=0.5),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        if dataset_properties is not None:
            for i in range(len(dataset_properties['categorical_columns'])
                           if isinstance(dataset_properties['categorical_columns'], List) else 0):
                ee_dimensions_search_space = HyperparameterSearchSpace(hyperparameter="dimension_reduction_" + str(i),
                                                                       value_range=dimension_reduction.value_range,
                                                                       default_value=dimension_reduction.default_value,
                                                                       log=dimension_reduction.log)
                add_hyperparameter(cs, ee_dimensions_search_space, UniformFloatHyperparameter)
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
