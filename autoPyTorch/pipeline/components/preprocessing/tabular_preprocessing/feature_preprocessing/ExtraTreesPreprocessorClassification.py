from typing import Any, Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.feature_preprocessing. \
    base_feature_preprocessor import autoPyTorchFeaturePreprocessingComponent
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter, check_none


class ExtraTreesPreprocessorClassification(autoPyTorchFeaturePreprocessingComponent):
    """
    Selects features based on importance weights calculated using extra trees
    """
    def __init__(self, bootstrap: bool = True, n_estimators: int = 10,
                 criterion: str = "gini", max_features: float = 0.5,
                 max_depth: Optional[Union[str, int]] = 5, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, min_weight_fraction_leaf: float = 0,
                 max_leaf_nodes: Optional[Union[str, int]] = "none",
                 min_impurity_decrease: float = 0, oob_score: bool = False,
                 verbose: int = 0,
                 random_state: Optional[np.random.RandomState] = None):
        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
        self.criterion = criterion
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.oob_score = oob_score
        self.verbose = verbose

        super().__init__(random_state=random_state)

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        elif isinstance(self.max_leaf_nodes, int):
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        else:
            raise ValueError(f"Expected `max_leaf_nodes` to be either "
                             f"in ('None', 'none', None) or an integer, got {self.max_leaf_nodes}")

        if check_none(self.max_depth):
            self.max_depth = None
        elif isinstance(self.max_depth, int):
            self.max_depth = int(self.max_depth)
        else:
            raise ValueError(f"Expected `max_depth` to be either "
                             f"in ('None', 'none', None) or an integer, got {self.max_depth}")

        # TODO: add class_weights
        estimator = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            oob_score=self.oob_score,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        self.preprocessor['numerical'] = SelectFromModel(estimator=estimator,
                                                         threshold='mean',
                                                         prefit=False)
        return self

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        bootstrap: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='bootstrap',
                                                                         value_range=(True, False),
                                                                         default_value=True,
                                                                         ),
        n_estimators: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='n_estimators',
                                                                            value_range=(10, 100),
                                                                            default_value=10,
                                                                            ),
        max_depth: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_depth',
                                                                         value_range=("none",),
                                                                         default_value="none",
                                                                         ),
        max_features: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_features',
                                                                            value_range=(0, 1),
                                                                            default_value=0.5,
                                                                            ),
        min_impurity_decrease: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter='min_impurity_decrease',
            value_range=(0,),
            default_value=0),
        criterion: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='criterion',
                                                                         value_range=("gini", "entropy"),
                                                                         default_value="gini",
                                                                         ),
        min_samples_split: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='min_samples_split',
                                                                                 value_range=(2, 20),
                                                                                 default_value=2,
                                                                                 ),
        min_samples_leaf: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='min_samples_leaf',
                                                                                value_range=(1, 20),
                                                                                default_value=1,
                                                                                ),
        min_weight_fraction_leaf: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter='min_weight_fraction_leaf',
            value_range=(0,),
            default_value=0),
        max_leaf_nodes: HyperparameterSearchSpace = HyperparameterSearchSpace(hyperparameter='max_leaf_nodes',
                                                                              value_range=("none",),
                                                                              default_value="none",
                                                                              ),
    ) -> ConfigurationSpace:

        cs = ConfigurationSpace()
        add_hyperparameter(cs, bootstrap, CategoricalHyperparameter)
        add_hyperparameter(cs, n_estimators, UniformIntegerHyperparameter)
        add_hyperparameter(cs, max_features, UniformFloatHyperparameter)
        add_hyperparameter(cs, min_impurity_decrease, UniformFloatHyperparameter)
        add_hyperparameter(cs, criterion, CategoricalHyperparameter)
        add_hyperparameter(cs, max_depth, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_samples_split, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_samples_leaf, UniformIntegerHyperparameter)
        add_hyperparameter(cs, min_weight_fraction_leaf, UniformFloatHyperparameter)
        add_hyperparameter(cs, max_leaf_nodes, UniformIntegerHyperparameter)

        return cs

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> Dict[str, Any]:
        return {'shortname': 'ETC',
                'name': 'Extra Trees Classifier Preprocessing',
                'handles_sparse': True,
                'handles_regression': False,
                'handles_classification': True
                }
