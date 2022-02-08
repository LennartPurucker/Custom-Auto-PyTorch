from functools import partial
from math import ceil, floor
from typing import Any, Callable, Dict, List, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.feature_selection import SelectPercentile, f_regression, mutual_info_regression
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class SelectPercentileRegression(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, score_func: str = "f_regression",
                 percentile: int = 50,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.percentile = percentile
        if score_func == "f_regression":
            self.score_func = f_regression
        elif score_func == "mutual_info":
            self.score_func = partial(mutual_info_regression, random_state=self.random_state)
        else:
            raise ValueError("score_func must be in ('f_regression', 'mutual_info'), "
                             "but is: %s" % score_func)

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = SelectPercentile(
            percentile=self.percentile, score_func=self.score_func)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        percentile: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="percentile",
                                                                          value_range=(1, 99),
                                                                          default_value=50,
                                                                          ),
        score_func: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="score_func",
                                                                          value_range=("f_regression", "mutual_info"),
                                                                          default_value="f_regression",
                                                                          ),
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        add_hyperparameter(cs, score_func, CategoricalHyperparameter)
        add_hyperparameter(cs, percentile, UniformIntegerHyperparameter)

        return cs


    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'SPR',
                'name': 'Select Percentile Regression',
                'handles_sparse': True,
                'handles_regression': True,
                'handles_classification': False
                }
