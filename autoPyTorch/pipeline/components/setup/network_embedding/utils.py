from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

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


def get_num_output_dimensions_reduced(config: Dict[str, Any], num_categs_per_feature: List[int]) -> List[int]:
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

    max_embedding_dim = config['max_embedding_dim']
    embed_exponent = config['embed_exponent']
    size_factor = config['embedding_size_factor']
    num_output_dimensions = [int(size_factor*max(
                                                 2,
                                                 min(max_embedding_dim,
                                                     1.6 * num_categories**embed_exponent)))
                             if num_categories > 2 else 1 for num_categories in num_categs_per_feature]
    return num_output_dimensions

def get_num_output_dimensions(config, embed_features, num_categories_per_col):
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

    num_output_dimensions = []

    embed_counter = 0
    for embed, num_category_per_col in zip(embed_features, num_categories_per_col):
        if embed:
            num_output_dimensions.append(ceil(config[f"dimension_reduction_{embed_counter}"] * num_category_per_col))
        else:
            num_output_dimensions.append(1)
        embed_counter += 1
    return num_output_dimensions

class _LearnedEntityEmbedding(nn.Module):
    """ Learned entity embedding module for categorical features"""

    def __init__(self, config: Dict[str, Any], num_categories_per_col: np.ndarray, num_features_excl_embed: int, reduced=False):
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
        self.embed_features = self.num_categories_per_col >= 2
        self.num_features_excl_embed = num_features_excl_embed

        if reduced:
            self.num_output_dimensions = get_num_output_dimensions_reduced(config, self.num_categories_per_col)
        else:
            self.num_output_dimensions = get_num_output_dimensions(
                config,
                self.embed_features,
                self.num_categories_per_col
            )

        self.num_out_feats = sum(self.num_output_dimensions)

        self.ee_layers = self._create_ee_layers()

    def get_partial_models(self, subset_features: List[int]) -> "_LearnedEntityEmbedding":
        """
        extract a partial models that only works on a subset of the data that ought to be passed to the embedding
        network, this function is implemented for time series forecasting tasks where the known future features is only
        a subset of the past features
        Args:
            subset_features (List[int]):
                a set of index identifying which features will pass through the partial model

        Returns:
            partial_model (_LearnedEntityEmbedding)
                a new partial model
        """
        num_input_features = self.num_categories_per_col[subset_features]
        num_features_excl_embed = sum([sf < self.num_features_excl_embed for sf in subset_features])

        num_output_dimensions = [self.num_output_dimensions[sf] for sf in subset_features]
        embed_features = [self.embed_features[sf] for sf in subset_features]

        ee_layers = []
        ee_layer_tracker = 0
        for sf in subset_features:
            if self.embed_features[sf]:
                ee_layers.append(self.ee_layers[ee_layer_tracker])
                ee_layer_tracker += 1
        ee_layers = nn.ModuleList(ee_layers)

        return PartialLearnedEntityEmbedding(num_input_features, num_features_excl_embed, embed_features,
                                             num_output_dimensions, ee_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pass the columns of each categorical feature through entity embedding layer
        # before passing it through the model
        concat_seq = []

        layer_pointer = 0
        for x_pointer, embed in enumerate(self.embed_features):
            current_feature_slice = x[:, x_pointer]
            if not embed:
                concat_seq.append(current_feature_slice.view(-1, 1))
                continue
            neg_indices = current_feature_slice < 0
            if neg_indices.any():
                warnings.warn(f"Negative category encountered in categorical feature: {layer_pointer}, setting to {self.max_category_per_col[x_pointer]}")
                # Set all neg_indices to max category
                current_feature_slice = torch.where(neg_indices, torch.tensor(self.max_category_per_col[x_pointer], device=x.device), current_feature_slice)
            current_feature_slice = current_feature_slice.to(torch.int)
            unique_cats = torch.unique(current_feature_slice)
            max_unique_cat = max(unique_cats)
            min_unique_cat = min(unique_cats)
            if min_unique_cat < 0:
                raise ValueError(
                    f"Negative category  {min_unique_cat} encountered in categorical feature: {x_pointer}")
            if max_unique_cat > self.num_categories_per_col[x_pointer]:
                raise ValueError(f"Category {max_unique_cat} encountered that is not in training data")
            concat_seq.append(self.ee_layers[layer_pointer](current_feature_slice))
            layer_pointer += 1

        return torch.cat(concat_seq, dim=1)

    def _create_ee_layers(self) -> nn.ModuleList:
        # entity embeding layers are Linear Layers
        layers = nn.ModuleList()
        self.max_category_per_col = []
        for num_cat, embed, num_out in zip(self.num_categories_per_col,
                                           self.embed_features,
                                           self.num_output_dimensions):
            if not embed:
                continue
            self.max_category_per_col.append(num_cat)
            layers.append(nn.Embedding(num_cat + 1, num_out))
        return layers


class PartialLearnedEntityEmbedding(_LearnedEntityEmbedding):
    """
    Construct a partial Embedding network that is derived from a learned embedding network and only applied to a subset
    of the input features. This is applied to forecasting tasks where not all the features might be known beforehand
    """
    def __init__(self,
                 num_categories_per_col: np.ndarray,
                 num_features_excl_embed: int,
                 embed_features: List[bool],
                 num_output_dimensions: List[int],
                 ee_layers: nn.Module
                 ):
        super(_LearnedEntityEmbedding, self).__init__()
        self.num_features_excl_embed = num_features_excl_embed
        # list of number of categories of categorical data
        # or 0 for numerical data
        self.num_categories_per_col = num_categories_per_col

        self.embed_features = embed_features

        self.num_output_dimensions = num_output_dimensions
        self.num_out_feats = self.num_features_excl_embed + sum(self.num_output_dimensions)

        self.ee_layers = ee_layers

        self.num_embed_features = self.num_categories_per_col[self.embed_features]
