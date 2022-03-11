
import logging.handlers
from typing import Dict, Optional, Union


import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.base_traditional_learner import \
    BaseCustomLearner
from autoPyTorch.pipeline.components.setup.predefined_models.custom_learners.knn.knn_utils import get_params as knn_get_params


class KNNModel(BaseCustomLearner):

    def __init__(self,
                 task_type: str,
                 output_type: str,
                 dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
                 optimize_metric: Optional[str] = None,
                 logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 random_state: Optional[np.random.RandomState] = None,
                 **kwargs
                 ):
        super(KNNModel, self).__init__(name="knn",
                                       logger_port=logger_port,
                                            random_state=random_state,
                                            task_type=task_type,
                                            output_type=output_type,
                                            optimize_metric=optimize_metric,
                                            dataset_properties=dataset_properties,
                                       params_func=knn_get_params)
        self.categoricals: Optional[np.ndarray[bool]] = None
        self.config.update(kwargs)

    def _preprocess(self,
                    X: np.ndarray
                    ) -> np.ndarray:

        super(KNNModel, self)._preprocess(X)
        if self.categoricals is None:
            self.categoricals = np.array([isinstance(X[0, ind], str) for ind in range(X.shape[1])])
        X = X[:, ~self.categoricals] if self.categoricals is not None else X

        return X

    def _prepare_model(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray
                       ) -> None:
        try:
            # TODO: Add more granular switch, currently this affects all future KNN models even if they had `use_daal=False`
            from sklearnex import patch_sklearn
            patch_sklearn("knn_classifier")
            patch_sklearn("knn_regressor")
            # sklearnex backend for KNN seems to be 20-40x+ faster than native sklearn with no downsides.
            self.logger.log(15, '\tUsing sklearnex KNN backend...')
        except:
            pass
        if not self.is_classification:
            self.model = KNeighborsRegressor(**self.config)
        else:
            self.num_classes = len(np.unique(y_train))
            # KNN is deterministic, no random seed needed
            self.model = KNeighborsClassifier(**self.config)

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
            'shortname': 'KNNLearner',
            'name': 'K Nearest Neighbors Learner',
        }
