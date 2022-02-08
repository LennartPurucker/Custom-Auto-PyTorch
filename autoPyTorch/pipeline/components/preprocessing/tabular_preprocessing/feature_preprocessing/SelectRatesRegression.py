from typing import Any, Dict, Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.feature_selection import GenericUnivariateSelect, f_regression

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class SelectPercentileRegression(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, score_func: str = "f_regression",
                 alpha: float = 0.1, mode: str = "fpr",
                 random_state: Optional[np.random.RandomState] = None
                 ):
        self.mode = mode
        self.alpha = alpha
        if score_func == "f_regression":
            self.score_func = f_regression
        else:
            raise ValueError("score_func must be in ('f_regression'), "
                             "but is: %s" % score_func)

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        self.preprocessor['numerical'] = GenericUnivariateSelect(
            mode=self.mode, score_func=self.score_func, param=self.alpha)

        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        alpha: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="alpha",
                                                                     value_range=(0.01, 0.5),
                                                                     default_value=0.1,
                                                                     ),
        mode: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="mode",
                                                                    value_range=('fpr', 'fdr', 'fwe'),
                                                                    default_value='fpr',
                                                                    ),
        score_func: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter="score_func",
                                                                          value_range=("f_regression",),
                                                                          default_value="f_regression",
                                                                          ),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()

        add_hyperparameter(cs, score_func, CategoricalHyperparameter)
        add_hyperparameter(cs, alpha, UniformFloatHyperparameter)
        add_hyperparameter(cs, mode, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'SRR',
                'name': 'Select Rates Regression',
                'handles_sparse': True,
                'handles_regression': True,
                'handles_classification': False
                }
