import logging.handlers
from typing import Dict, Optional, Union

import logging


from lightgbm import LGBMClassifier, LGBMRegressor

import numpy as np

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.base_traditional_learner import \
    BaseTraditionalLearner
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.lgbm.utils import early_stopping_custom, get_metric, get_params as lgb_get_params, get_train_loss_name



class LGBModel(BaseTraditionalLearner):
    def __init__(self,
                 task_type: str,
                 output_type: str,
                 dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None,
                 **kwargs
                 ):
        super(LGBModel, self).__init__(name="lgb",
                                       logger_port=logger_port,
                                            random_state=random_state,
                                            task_type=task_type,
                                            output_type=output_type,
                                            optimize_metric=optimize_metric,
                                            dataset_properties=dataset_properties,
                                       params_func=lgb_get_params)
        self.config.update(kwargs)

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        early_stopping = 150 if X_train.shape[0] > 10000 else max(round(150 * 10000 / X_train.shape[0]), 10)
        self.config["early_stopping_rounds"] = early_stopping
        self.stopping_metric_name = get_metric(output_type=self.output_type, optimize_metric=self.metric.name)
        self.training_objective = get_train_loss_name(self.output_type)
        if not self.is_classification:
            self.model = LGBMRegressor(**self.config, random_state=self.random_state)
        else:
            self.num_classes = len(np.unique(y_train)) if len(np.unique(y_train)) != 2 else 1  # this fixes a bug
            self.config["num_class"] = self.num_classes

            self.model = LGBMClassifier(**self.config, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray
             ) -> None:
        assert self.model is not None, "No model found. Can't fit without preparing the model"

        callbacks = [
            # TODO: pass start time and time limit to early stopping
            early_stopping_custom(self.config["early_stopping_rounds"], logger=self.logger, metrics_to_use=[('valid_set', self.stopping_metric_name)], max_diff=None,  # , start_time=start_time, time_limit=time_limit,
                                  ignore_dart_warning=True, verbose=False, manual_stop_file=False, train_loss_name=self.training_objective),
        ]
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=self.training_objective, callbacks=callbacks)

    def predict(self, X_test: np.ndarray,
                predict_proba: bool = False,
                preprocess: bool = True) -> np.ndarray:
        assert self.model is not None, "No model found. Can't " \
                                       "predict before fitting. " \
                                       "Call fit before predicting"
        if preprocess:
            X_test = self._preprocess(X_test)

        if predict_proba:
            if not self.is_classification:
                raise ValueError("Can't predict probabilities for a regressor")
            y_pred_proba = self.model.predict_proba(X_test)
            if self.num_classes == 2:
                y_pred_proba = y_pred_proba.transpose()[0:len(X_test)]
            return y_pred_proba

        y_pred = self.model.predict(X_test)
        return y_pred

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'LGBMLearner',
            'name': 'Light Gradient Boosting Machine Learner',
        }