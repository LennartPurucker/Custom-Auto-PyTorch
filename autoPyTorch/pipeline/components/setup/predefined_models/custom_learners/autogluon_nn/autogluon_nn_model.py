from gc import callbacks
import logging.handlers
import tempfile
from typing import Dict, Optional, Union


import numpy as np

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.base_traditional_learner import \
    BaseCustomLearner
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.catboost.catboost_utils import (
    get_params
)


class AutogluonNNModel(BaseCustomLearner):
    def __init__(self,
                 task_type: str,
                 output_type: str,
                 optimize_metric: Optional[str] = None,
                 dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None,
                 **kwargs
                 ):
        super(AutogluonNNModel, self).__init__(name="neural_network",
                                               logger_port=logger_port,
                                               random_state=random_state,
                                               task_type=task_type,
                                               output_type=output_type,
                                               optimize_metric=optimize_metric,
                                               dataset_properties=dataset_properties,
                                               params_func=get_params)
        self.config["train_dir"] = tempfile.gettempdir()
        self.config.update(kwargs)

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        if not self.is_classification:
            self.config['eval_metric'] = AutoPyTorchToCatboostMetrics[self.metric.name].value
            # CatBoost Cannot handle a random state object, just the seed
            self.model = CatBoostRegressor(**self.config, random_state=self.random_state.get_state()[1][0])
        else:
            self.config['eval_metric'] = AutoPyTorchToCatboostMetrics[self.metric.name].value
            # CatBoost Cannot handle a random state object, just the seed
            self.model = CatBoostClassifier(**self.config, random_state=self.random_state.get_state()[1][0])

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray) -> None:

        assert self.model is not None, "No model found. Can't fit without preparing the model"
        early_stopping = 150 if X_train.shape[0] > 10000 else max(round(150 * 10000 / X_train.shape[0]), 10)
        callbacks = []
        callbacks.append(EarlyStoppingCallback(stopping_rounds=early_stopping, eval_metric=self.config['eval_metric']))
        num_rows_train = X_train.shape[0]
        num_cols_train = X_train.shape[1]
        self.num_classes = len(np.unique(y_train)) if len(np.unique(y_train)) != 2 else 1
        if num_rows_train * num_cols_train * self.num_classes > 5_000_000:
            # The data is large enough to potentially cause memory issues during training, so monitor memory usage via callback.
            callbacks.append(MemoryCheckCallback())
        categoricals = [ind for ind in range(X_train.shape[1]) if isinstance(X_train[0, ind], str)]

        X_train_pooled = Pool(data=X_train, label=y_train, cat_features=categoricals)
        X_val_pooled = Pool(data=X_val, label=y_val, cat_features=categoricals)

        self.model.fit(X_train_pooled,
                       eval_set=X_val_pooled,
                       use_best_model=True,
                       early_stopping_rounds=early_stopping,
                       callbacks=callbacks,
                       verbose=False)

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'CBLearner',
            'name': 'Categorical Boosting Learner',
        }
