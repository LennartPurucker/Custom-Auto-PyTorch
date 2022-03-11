import logging.handlers
from typing import Dict, Optional, Union


import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.svm import SVC, SVR

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.base_traditional_learner import \
    BaseCustomLearner


class SVMModel(BaseCustomLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None
                 ):
        super(SVMModel, self).__init__(name="svm",
                                       logger_port=logger_port,
                                       random_state=random_state,
                                       task_type=task_type,
                                       output_type=output_type,
                                       optimize_metric=optimize_metric)

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        if not self.is_classification:
            # Does not take random state.
            self.model = SVR(**self.config)
        else:
            self.model = SVC(**self.config, probability=True, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"
        self.model.fit(X_train, y_train)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'SVMLearner',
            'name': 'Support Vector Machine Learner',
        }
