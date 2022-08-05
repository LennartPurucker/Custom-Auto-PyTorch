from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np

from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin

import torch

import sys
import warnings

from typing import Any, Callable, Dict, List, Optional, Union
import typing

import joblib
import numpy as np
import pandas as pd

from autoPyTorch.constants import REGRESSION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.ensemble.ensemble_selection_types import BaseLayerEnsembleSelectionTypes, StackingEnsembleSelectionTypes
from autoPyTorch.utils.logging_ import PicklableClientLogger


def get_loss_weight_strategy(loss: Type[torch.nn.Module]) -> Callable:
    """
    Utility function that returns strategy for the given loss
    Args:
        loss (Type[torch.nn.Module]): type of the loss function
    Returns:
        (Callable): Relevant Callable strategy
    """
    if loss.__name__ in LossWeightStrategyWeightedBinary.get_properties()['supported_losses']:
        return LossWeightStrategyWeightedBinary()
    elif loss.__name__ in LossWeightStrategyWeighted.get_properties()['supported_losses']:
        return LossWeightStrategyWeighted()
    else:
        raise ValueError("No strategy currently supports the given loss, {}".format(loss.__name__))


class LossWeightStrategyWeighted():
    def __call__(self, y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy() if y.is_cuda else y.numpy()
        counts = np.sum(y, axis=0)
        total_weight = y.shape[0]

        if len(y.shape) > 1 and y.shape[1] != 1:
            # In this case, the second axis represents classes
            weight_per_class = total_weight / y.shape[1]
            weights = (np.ones(y.shape[1]) * weight_per_class) / np.maximum(counts, 1)
        else:
            # Numpy unique return the sorted classes. This is desirable as
            # weights recieved by PyTorch is a sorted list of classes
            classes, counts = np.unique(y, axis=0, return_counts=True)
            weight_per_class = total_weight / classes.shape[0]
            weights = (np.ones(classes.shape[0]) * weight_per_class) / counts

        return weights

    @staticmethod
    def get_properties() -> Dict[str, Any]:
        return {'supported_losses': ['CrossEntropyLoss']}


class LossWeightStrategyWeightedBinary():
    def __call__(self, y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy() if y.is_cuda else y.numpy()
        counts_one = np.sum(y, axis=0)
        counts_zero = y.shape[0] - counts_one
        weights = counts_zero / np.maximum(counts_one, 1)

        return np.array(weights)

    @staticmethod
    def get_properties() -> Dict[str, Any]:
        return {'supported_losses': ['BCEWithLogitsLoss']}


class MinorityCoalesceTransformer(BaseEstimator, TransformerMixin):
    """ Group together categories whose occurrence is less than a specified min_frac."""
    def __init__(self, min_frac: Optional[float] = None):
        self.min_frac = min_frac
        self._categories_to_coalesce: Optional[List[np.ndarray]] = None

        if self.min_frac is not None and (self.min_frac < 0 or self.min_frac > 1):
            raise ValueError(f"min_frac for {self.__class__.__name__} must be in [0, 1], but got {min_frac}")

    def _check_dataset(self, X: Union[np.ndarray, sparse.csr_matrix]) -> None:
        """
        When transforming datasets, we modify values to:
            *  0 for nan values
            * -1 for unknown values
            * -2 for values to be coalesced
        For this reason, we need to check whether datasets have values
        smaller than -2 to avoid mis-transformation.
        Note that zero-imputation is the default setting in SimpleImputer of sklearn.

        Args:
            X (np.ndarray):
                The input features from the user, likely transformed by an encoder and imputator.
        """
        X_data = X.data if sparse.issparse(X) else X
        if np.nanmin(X_data) <= -2:
            raise ValueError("The categoricals in input features for MinorityCoalesceTransformer "
                             "cannot have integers smaller than -2.")

    @staticmethod
    def _get_column_data(
        X: Union[np.ndarray, sparse.csr_matrix],
        col_idx: int,
        is_sparse: bool
    ) -> Union[np.ndarray, sparse.csr_matrix]:
        """
        Args:
            X (Union[np.ndarray, sparse.csr_matrix]):
                The feature tensor with only categoricals.
            col_idx (int):
                The index of the column to get the data.
            is_sparse (bool):
                Whether the tensor is sparse or not.

        Return:
            col_data (Union[np.ndarray, sparse.csr_matrix]):
                The column data of the tensor.
        """

        if is_sparse:
            assert not isinstance(X, np.ndarray)  # mypy check
            indptr_start = X.indptr[col_idx]
            indptr_end = X.indptr[col_idx + 1]
            col_data = X.data[indptr_start:indptr_end]
        else:
            col_data = X[:, col_idx]

        return col_data

    def fit(self, X: Union[np.ndarray, sparse.csr_matrix],
            y: Optional[np.ndarray] = None) -> 'MinorityCoalesceTransformer':
        """
        Train the estimator to identify low frequency classes on the input train data.

        Args:
            X (Union[np.ndarray, sparse.csr_matrix]):
                The input features from the user, likely transformed by an encoder and imputator.
            y (Optional[np.ndarray]):
                Optional labels for the given task, not used by this estimator.
        """
        self._check_dataset(X)
        n_instances, n_features = X.shape

        if self.min_frac is None:
            self._categories_to_coalesce = [np.array([]) for _ in range(n_features)]
            return self

        categories_to_coalesce: List[np.ndarray] = []
        is_sparse = sparse.issparse(X)
        for col in range(n_features):
            col_data = self._get_column_data(X=X, col_idx=col, is_sparse=is_sparse)
            unique_vals, counts = np.unique(col_data, return_counts=True)
            frac = counts / n_instances
            categories_to_coalesce.append(unique_vals[frac < self.min_frac])

        self._categories_to_coalesce = categories_to_coalesce
        return self

    def transform(
        self,
        X: Union[np.ndarray, sparse.csr_matrix]
    ) -> Union[np.ndarray, sparse.csr_matrix]:
        """
        Coalesce categories with low frequency in X.

        Args:
            X (Union[np.ndarray, sparse.csr_matrix]):
                The input features from the user, likely transformed by an encoder and imputator.
        """
        self._check_dataset(X)

        if self._categories_to_coalesce is None:
            raise RuntimeError("fit() must be called before transform()")

        if self.min_frac is None:
            return X

        n_features = X.shape[1]
        is_sparse = sparse.issparse(X)

        for col in range(n_features):
            # -2 stands coalesced. For more details, see the doc in _check_dataset
            col_data = self._get_column_data(X=X, col_idx=col, is_sparse=is_sparse)
            mask = np.isin(col_data, self._categories_to_coalesce[col])
            col_data[mask] = -2

        return X

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)



def _pipeline_predict(pipeline,
                      X: Union[np.ndarray, pd.DataFrame],
                      batch_size: int,
                      logger: PicklableClientLogger,
                      task: int) -> np.ndarray:
    @typing.no_type_check
    def send_warnings_to_log(
            message, category, filename, lineno, file=None, line=None):
        logger.debug('%s:%s: %s:%s' % (filename, lineno, category.__name__, message))
        return

    X_ = X.copy()
    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        if task in REGRESSION_TASKS:
            # Voting regressor does not support batch size
            prediction = pipeline.predict(X_)
        else:
            # Voting classifier predict proba does not support batch size
            prediction = pipeline.predict_proba(X_)
            # Check that all probability values lie between 0 and 1.
            if not ((prediction >= 0).all() and (prediction <= 1).all()):
                np.set_printoptions(threshold=sys.maxsize)
                raise ValueError("For {}, prediction probability not within [0, 1]: {}/{}!".format(
                    pipeline,
                    prediction,
                    np.sum(prediction, axis=1)
                ))

    if len(prediction.shape) < 1 or len(X_.shape) < 1 or \
            X_.shape[0] < 1 or prediction.shape[0] != X_.shape[0]:
        logger.warning(
            "Prediction shape for model %s is %s while X_.shape is %s",
            pipeline, str(prediction.shape), str(X_.shape)
        )
    return prediction


def predict_with_stacked_ensemble(
        X_test,
        batch_size,
        n_jobs,
        models,
        ensemble_identifiers,
        logger,
        task_type,
        base_ensemble_method,
        stacking_ensemble_method,
        ensemble
    ):
    X_test_copy = X_test.copy()

    stacked_ensemble_predictions = []
    for i, (model, layer_identifiers) in enumerate(zip(models, ensemble_identifiers)):
        nonnull_identifiers = [identifier for identifier in layer_identifiers if identifier is not None]
        if len(nonnull_identifiers) == 0:
            break

        logger.debug(f"layer : {i} of stacking ensemble,\n layer identifiers: {layer_identifiers},\n model: {model}")
        all_predictions = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(_pipeline_predict)(
                    model[identifier], X_test_copy, batch_size, logger, STRING_TO_TASK_TYPES[task_type]
                )
                for identifier in nonnull_identifiers
            )
        if (
            base_ensemble_method in (BaseLayerEnsembleSelectionTypes.ensemble_autogluon, BaseLayerEnsembleSelectionTypes.ensemble_selection)
                or stacking_ensemble_method == StackingEnsembleSelectionTypes.stacking_repeat_models
        ):
            concat_all_predictions = ensemble.get_expanded_layer_stacking_ensemble_predictions(
                    stacking_layer=i, raw_stacking_layer_ensemble_predictions=all_predictions)
        else:
            concat_all_predictions = all_predictions

        stacked_ensemble_predictions.append(all_predictions)

        X_test_copy = np.concatenate([X_test, *concat_all_predictions], axis=1)
        logger.debug(f"shape of X_test after predict with layer : {i} = {X_test_copy.shape}")

    identifier_to_predictions = {}
    for layer_identifiers, layer_predictions in zip(ensemble_identifiers, stacked_ensemble_predictions):
        for identifier, pred in zip(layer_identifiers, layer_predictions):
            identifier_to_predictions[identifier] = pred

    return identifier_to_predictions, all_predictions
