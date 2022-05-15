import logging.handlers
import tempfile
from typing import Dict, Optional, Union

import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from autoPyTorch.constants import MULTICLASS
from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.base_traditional_learner import \
    BaseTraditionalLearner
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.random_forest.utils import get_params


class RFModel(BaseTraditionalLearner):
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
        super(RFModel, self).__init__(name="random_forest",
                                      logger_port=logger_port,
                                      random_state=random_state,
                                      task_type=task_type,
                                      output_type=output_type,
                                      optimize_metric=optimize_metric,
                                      dataset_properties=dataset_properties,
                                      time_limit=time_limit,
                                      params_func=get_params)
        self.config.update(kwargs)

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:

        self.config["warm_start"] = False
        # TODO: Check if we need to warmstart for regression.
        #  In autogluon, they warm start when usinf daal backend, see
        #  ('https://github.com/awslabs/autogluon/blob/master/tabular/src/autogluon/tabular/models/rf/rf_model.py#L35')
        if not self.is_classification:
            self.model = RandomForestRegressor(**self.config, random_state=self.random_state)
        else:
            self.num_classes = len(np.unique(y_train))
            if self.num_classes > 2:
                self.logger.info("==> Using warmstarting for multiclass")
                self.final_n_estimators = self.config["n_estimators"]
                self.config["n_estimators"] = 8
                self.config["warm_start"] = True
            self.model = RandomForestClassifier(**self.config, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"

        self.model = self.model.fit(X_train, y_train)
        if self.config["warm_start"]:
            self.model.n_estimators = self.final_n_estimators
            self.model.fit(X_train, y_train)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'RFLearner',
            'name': 'Random Forest Learner',
        }
