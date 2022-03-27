import logging.handlers
from typing import Dict, Optional, Union


import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.base_traditional_learner import \
    BaseCustomLearner
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.extratrees.extratrees_utils import get_params


class ExtraTreesModel(BaseCustomLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None,
                 **kwargs
                 ):
        super(ExtraTreesModel, self).__init__(name="extra_trees",
                                              logger_port=logger_port,
                                              random_state=random_state,
                                              task_type=task_type,
                                              output_type=output_type,
                                              optimize_metric=optimize_metric,
                                              dataset_properties=dataset_properties,
                                              params_func=get_params)

    def _prepare_model(
        self,
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
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'ETLearner',
            'name': 'ExtraTreesLearner',
        }
