from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, check_none


class LibLinearSVCPreprocessor(autoPyTorchFeaturePreprocessingComponent):
    def __init__(self, dual: bool = False, penalty: str = "l1",
                 loss: str = "squared_hinge", tol: float = 1e-4,
                 C: float = 1, multi_class: str = "ovr",
                 intercept_scaling: int = 1, fit_intercept: bool = True,
                 random_state: Optional[np.random.RandomState] = None):

        self.dual = dual
        self.penalty = penalty
        self.loss = loss
        self.multi_class = multi_class
        self.intercept_scaling = intercept_scaling
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.C = C

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        # TODO: add class_weights
        estimator = LinearSVC(penalty=self.penalty,
                              loss=self.loss,
                              dual=self.dual,
                              tol=self.tol,
                              C=self.C,
                              fit_intercept=self.fit_intercept,
                              intercept_scaling=self.intercept_scaling,
                              multi_class=self.multi_class,
                              random_state=self.random_state
                              )

        self.preprocessor['numerical'] = SelectFromModel(estimator=estimator,
                                                         threshold='mean',
                                                         prefit=False)
        return self


    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'LinearSVC Preprocessor',
                'name': 'Liblinear Support Vector Classification Preprocessing',
                'handles_sparse': True,
                'handles_classification': True,
                'handles_regression': False
                }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        penalty: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='penalty',
                                                                         value_range=("l1",),
                                                                         default_value="l1",
                                                                         ),
        loss: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='loss',
                                                                    value_range=("hinge", "squared_hinge"),
                                                                    default_value="squared_hinge",
                                                                    ),
        dual: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='dual',
                                                                    value_range=(False,),
                                                                    default_value=False,
                                                                    ),
        tol: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='tol',
                                                                                 value_range=(1e-5, 1e-1),
                                                                                 default_value=1e-4,
                                                                                 log=True
                                                                                 ),
        C: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='C',
                                                                                value_range=(0.03125, 32768),
                                                                                default_value=1,
                                                                                log=True
                                                                                ),
        multi_class: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='multi_class',
                                                                           value_range=("ovr",),
                                                                           default_value="ovr"),
        fit_intercept: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='fit_intercept',
                                                                              value_range=(True,),
                                                                              default_value=True,
                                                                              ),
        intercept_scaling: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='intercept_scaling',
                                                                                 value_range=(1,),
                                                                                 default_value=1,
                                                                                 ),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, fit_intercept, CategoricalHyperparameter)
        add_hyperparameter(cs, penalty, CategoricalHyperparameter)
        add_hyperparameter(cs, multi_class, CategoricalHyperparameter)
        add_hyperparameter(cs, loss, CategoricalHyperparameter)
        add_hyperparameter(cs, dual, CategoricalHyperparameter)
        add_hyperparameter(cs, tol, UniformFloatHyperparameter)
        add_hyperparameter(cs, C, UniformFloatHyperparameter)
        add_hyperparameter(cs, intercept_scaling, UniformIntegerHyperparameter)

        return cs
