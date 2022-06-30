import logging.handlers
from typing import Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from autoPyTorch.constants import CLASSIFICATION_TASKS, REGRESSION_TASKS, STRING_TO_TASK_TYPES

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.base_traditional_learner import \
    BaseTraditionalLearner
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.extratrees.utils import get_params
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class ExtraTreesModel(BaseTraditionalLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None,
                 time_limit: Optional[int] = None,
                 **kwargs
                 ):
        super(ExtraTreesModel, self).__init__(name="extra_trees",
                                              logger_port=logger_port,
                                              random_state=random_state,
                                              task_type=task_type,
                                              output_type=output_type,
                                              optimize_metric=optimize_metric,
                                              dataset_properties=dataset_properties,
                                              time_limit=time_limit,
                                              params_func=get_params)
                                              
                                              
    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        self.config["warm_start"] = False

        if not self.is_classification:
            self.model = ExtraTreesRegressor(**self.config, random_state=self.random_state)
        else:
            self.num_classes = len(np.unique(y_train))
            if self.num_classes > 2:
                self.logger.info("==> Using warmstarting for multiclass")
                self.final_n_estimators = self.config["n_estimators"]
                self.config["n_estimators"] = 8
                self.config["warm_start"] = True

            self.model = ExtraTreesClassifier(**self.config, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"
        self.model.fit(X_train, y_train)
        if self.config["warm_start"]:
            self.model.n_estimators = self.final_n_estimators
            self.model.fit(X_train, y_train)

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        criterion: HyperparameterSearchSpace = HyperparameterSearchSpace('criterion',
                                                                        value_range=['gini', 'entropy', 'squared_error'],
                                                                        default_value='gini'),
    ) -> ConfigurationSpace:
        """Get the hyperparameter search space for the SimpleImputer

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]])
                Properties that describe the dataset
                Note: Not actually Optional, just adhering to its supertype
            numerical_strategy (HyperparameterSearchSpace: default = ...)
                The strategy to use for numerical imputation

        Returns:
            ConfigurationSpace
                The space of possible configurations for a SimpleImputer with the given
                `dataset_properties`
        """
        cs = ConfigurationSpace()

        if dataset_properties is not None:
            if STRING_TO_TASK_TYPES[dataset_properties['task_type']] in CLASSIFICATION_TASKS:
                if 'squared_error' in criterion.value_range:
                    value_range = [value for value in criterion.value_range if value != 'squared_error']
                    default_value = value_range[0]
                    criterion = HyperparameterSearchSpace(criterion.hyperparameter,
                                                          value_range=value_range,
                                                          default_value=default_value)
            elif STRING_TO_TASK_TYPES[dataset_properties['task_type']] in REGRESSION_TASKS:
                criterion = HyperparameterSearchSpace(criterion.hyperparameter,
                                                        value_range=('squared_error',),
                                                        default_value='squared_error')
        add_hyperparameter(cs, criterion, CategoricalHyperparameter)
        return cs

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ETLearner',
            'name': 'ExtraTreesLearner',
        }
