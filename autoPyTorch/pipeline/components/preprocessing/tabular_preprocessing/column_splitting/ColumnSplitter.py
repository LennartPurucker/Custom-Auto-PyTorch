from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter
)

import pandas as pd

import numpy as np

from scipy.stats import skew

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import \
    autoPyTorchTabularPreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, ispandas


def _get_skew(
    data: Union[np.ndarray, pd.DataFrame]
)->float:
    return data.skew() if ispandas(data) else skew(data)

class ColumnSplitter(autoPyTorchTabularPreprocessingComponent):
    """
    Removes features that have the same value in the training data.
    """
    def __init__(
        self,
        min_categories_for_embedding: float = 5,
        skew_threshold: float = 0.99,
        random_state: Optional[np.random.RandomState] = None
    ):
        self.min_categories_for_embedding = min_categories_for_embedding
        self.skew_threshold = skew_threshold

        self.special_feature_types = dict(skew_columns=[], encode_columns=[], embed_columns=[], scale_columns=[])
        self.num_categories_per_col: Optional[List] = None
        super().__init__()

    def fit(self, X: Dict[str, Any], y: Optional[Any] = None) -> 'ColumnSplitter':

        self.check_requirements(X, y)

        if len(X['dataset_properties']['categorical_columns']) > 0:
            self.num_categories_per_col = []
        for categories_per_column, column in zip(X['dataset_properties']['num_categories_per_col'], X['dataset_properties']['categorical_columns']):
            if (
                categories_per_column >= self.min_categories_for_embedding
            ):
                self.special_feature_types['embed_columns'].append(column)
                self.num_categories_per_col.append(categories_per_column)
            else:
                self.special_feature_types['encode_columns'].append(column)

        # Make sure each column is a valid type
        for column in X['dataset_properties']['numerical_columns']:
            
                if np.abs(_get_skew(X['X_train'][X['train_indices']][column])) > self.skew_threshold:
                    self.special_feature_types['skew_columns'].append(column)
                else:
                    self.special_feature_types['scale_columns'].append(column)

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        if self.num_categories_per_col is not None:
            X['dataset_properties']['num_categories_per_col'] = self.num_categories_per_col
        X.update(self.special_feature_types)
        return X

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:

        return {
            'shortname': 'ColumnSplitter',
            'name': 'Column Splitter',
            'handles_sparse': False,
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        min_categories_for_embedding: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter="min_categories_for_embedding",
            value_range=(3, 7),
            default_value=3,
            log=True),
        skew_threshold: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="skew_threshold",
                                                                                   value_range=(0.1, 1),
                                                                                   default_value=0.99,
                                                                                   log=True
                                                                                   )
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, min_categories_for_embedding, UniformIntegerHyperparameter)
        add_hyperparameter(cs, skew_threshold, UniformFloatHyperparameter)

        return cs
