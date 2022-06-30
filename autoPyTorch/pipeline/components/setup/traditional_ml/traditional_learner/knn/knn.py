import logging.handlers
from typing import Dict, Optional, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

import numpy as np

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


from autoPyTorch.pipeline.base_pipeline import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.base_traditional_learner import \
    BaseTraditionalLearner
from autoPyTorch.pipeline.components.setup.traditional_ml.traditional_learner.knn.utils import get_params as knn_get_params
from autoPyTorch.utils.common import HyperparameterSearchSpace, add_hyperparameter


class KNNModel(BaseTraditionalLearner):

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
        super(KNNModel, self).__init__(name="knn",
                                       logger_port=logger_port,
                                       random_state=random_state,
                                       task_type=task_type,
                                       output_type=output_type,
                                       optimize_metric=optimize_metric,
                                       dataset_properties=dataset_properties,
                                       time_limit=time_limit,
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
    def get_hyperparameter_search_space(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        weights: HyperparameterSearchSpace = HyperparameterSearchSpace('weights',
                                                                        value_range=['uniform', 'distance'],
                                                                        default_value='uniform')
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

        add_hyperparameter(cs, weights, CategoricalHyperparameter)

        return cs

    @staticmethod
    def get_properties(
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
    ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'KNNLearner',
            'name': 'K Nearest Neighbors Learner',
        }
