import logging.handlers
from time import time
from typing import Dict, Optional, Union

import logging


from lightgbm import LGBMClassifier, LGBMRegressor

import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.base_traditional_learner import \
    BaseTraditionalLearner
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.xgboost.utils import get_metric, get_param_baseline as xgb_get_params
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.xgboost.early_stopping_custom import EarlyStoppingCustom
from autoPyTorch.utils.early_stopping import get_early_stopping_rounds



class XGBModel(BaseTraditionalLearner):
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
        super(XGBModel, self).__init__(name="lgb",
                                       logger_port=logger_port,
                                       random_state=random_state,
                                       task_type=task_type,
                                       output_type=output_type,
                                       optimize_metric=optimize_metric,
                                       dataset_properties=dataset_properties,
                                       time_limit=time_limit,
                                       params_func=xgb_get_params)
        self.config.update(kwargs)
        self.encoder = None

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        from xgboost import XGBClassifier, XGBRegressor
        self.eval_metric = get_metric(self.output_type, optimize_metric=self.metric.name)
        # avoid unnecessary warnings
        self.config['eval_metric'] = get_metric(self.output_type, optimize_metric=self.metric.name)
        if not self.is_classification:
            self.model = XGBRegressor(**self.config, random_state=self.random_state)
        else:
            self.config["num_class"] = len(np.unique(y_train)) if len(np.unique(y_train)) != 2 else 1  # this fixes a bug

            self.model = XGBClassifier(**self.config, random_state=self.random_state)

    def _fit(self, X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray
             ) -> None:
        start_time = time()
     
        assert self.model is not None, "No model found. Can't fit without preparing the model"
        eval_set = []
        if X_val is None:
            early_stopping_rounds = None
            eval_set = None
        else:
            eval_set.append((X_val, y_val))
            early_stopping_rounds = get_early_stopping_rounds(X_train.shape[0])

        callbacks = []
        if eval_set is not None:
            callbacks.append(EarlyStoppingCustom(early_stopping_rounds, start_time=start_time, time_limit=self.time_limit))
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=self.eval_metric, callbacks=callbacks, verbose=False)

    def _preprocess(self,
                    X: np.ndarray
                    ) -> np.ndarray:

        super(XGBModel, self)._preprocess(X)

        if len(self.dataset_properties['categorical_columns']) > 0:
            if self.encoder is None:
                self.encoder = make_column_transformer((OneHotEncoder(sparse=False, handle_unknown='ignore'), self.dataset_properties['categorical_columns']), remainder="passthrough")
                self.encoder.fit(X)
            X = self.encoder.transform(X)   

        return X

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
            'shortname': 'XGBLearner',
            'name': 'Xtreme Gradient Boosting Machine Learner',
        }