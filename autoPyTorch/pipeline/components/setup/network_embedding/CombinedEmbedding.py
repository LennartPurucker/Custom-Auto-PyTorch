import math
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
)

from math import ceil

import numpy as np

import torch
from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.network_embedding.base_network_embedding import NetworkEmbeddingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter

def get_num_output_dimensions(config, num_embed_features):
    """
        Returns list of embedding sizes for each categorical variable.
        Selects this adaptively based on training_datset.
        Note: Assumes there is at least one embed feature.
    Args:
        config (Dict[str, Any]): 
            contains the hyperparameters required to calculate the `num_output_dimensions`
        num_categs_per_feature (List[int]):
            list containing number of categories for each feature that is to be embedded,
            0 if the column is not an embed column
    Returns:
        List[int]:
            list containing the output embedding size for each column,
            1 if the column is not an embed column
    """

    return [config["embedding_dim"] * num_embed_features.shape[0]]

class _CombinedEmbedding(nn.Module):
    """ Learned entity embedding module for categorical features"""

    def __init__(self, config: Dict[str, Any], num_categories_per_col: np.ndarray, num_features_excl_embed: int):
        """
        Args:
            config (Dict[str, Any]): The configuration sampled by the hyperparameter optimizer
            num_categories_per_col (np.ndarray): number of categories per categorical columns that will be embedded
            num_features_excl_embed (int): number of features in X excluding the features that need to be embedded
        """
        super().__init__()
        self.config = config
        # list of number of categories of categorical data
        # or 0 for numerical data
        self.num_categories_per_col = num_categories_per_col
        self.embed_features = np.array(self.num_categories_per_col > 0)
        self.num_features_excl_embed = num_features_excl_embed

        self.num_embed_features = self.num_categories_per_col[self.embed_features]

        category_offsets = torch.tensor([0] + self.num_embed_features[:-1].tolist()).cumsum(0)
        self.register_buffer("category_offsets", category_offsets)
        self.max_category = int(sum(self.num_embed_features))
        self.category_embeddings = nn.Embedding(self.max_category + 1, config["embedding_dim"])
        nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        
        self.num_output_dimensions = get_num_output_dimensions(
            config,
            self.num_embed_features
        )

        self.num_out_feats = num_features_excl_embed + sum(self.num_output_dimensions)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass the columns of each categorical feature through entity embedding layer
        # before passing it through the model
        x_num = x[:, ~self.embed_features]
        x_cat = x[:, self.embed_features].long()
        neg_indices = x_cat < 0
        x_cat = x_cat + self.category_offsets
        if neg_indices.any():
            warnings.warn(
                f"Negative indices in input to embedding: {neg_indices}. Setting value to max category: {self.max_category}."
            )
            x_cat[neg_indices] = self.max_category
        x_cat = self.category_embeddings(x_cat).view(x_cat.size(0), -1)
        return torch.cat([x_num, x_cat], dim=1)


class CombinedEmbedding(NetworkEmbeddingComponent):
    """
    Class to learn an embedding for categorical hyperparameters.
    """

    def __init__(self, random_state: Optional[np.random.RandomState] = None, **kwargs: Any):
        super().__init__(random_state=random_state)
        self.config = kwargs

    def build_embedding(self, num_categories_per_col: np.ndarray, num_features_excl_embed: int) -> Tuple[_CombinedEmbedding, int]:

        embedding = _CombinedEmbedding(config=self.config,
                                       num_categories_per_col=num_categories_per_col,
                                       num_features_excl_embed=num_features_excl_embed)
        self.num_out_feats = embedding.num_out_feats
        return embedding, embedding.num_output_dimensions

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, Any]] = None,
        dimension_reduction: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="embedding_dim",
            value_range=(64, 512),
            default_value=128),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        if dataset_properties is not None:
            if 'num_categorical_columns' in dataset_properties:
                if dataset_properties['num_categorical_columns'] == 0:
                    return cs
        add_hyperparameter(cs, dimension_reduction, UniformIntegerHyperparameter)
        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'embedding',
            'name': 'Comined Embedding',
            'handles_tabular': True,
            'handles_image': False,
            'handles_time_series': False,
        }
