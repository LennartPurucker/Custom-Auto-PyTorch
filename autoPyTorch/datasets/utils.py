from typing import Dict, List, Optional

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import ResamplingStrategies
from autoPyTorch.constants import (
    STRING_TO_TASK_TYPES,
    CLASSIFICATION_TASKS,
)
from autoPyTorch.utils.data_classes import get_data_validator_class, get_dataset_class


def get_appended_dataset(
    original_dataset: BaseDataset,
    previous_layer_predictions_train: List[Optional[np.ndarray]],
    previous_layer_predictions_test: List[Optional[np.ndarray]],
    resampling_strategy: ResamplingStrategies,
    resampling_strategy_args: Optional[Dict]):
    X_train, y_train = original_dataset.train_tensors
    X_test, y_test = original_dataset.test_tensors

    
    X_train =  np.concatenate([X_train, *previous_layer_predictions_train], axis=1)
    X_test = np.concatenate([X_test, *previous_layer_predictions_test], axis=1)

    new_feat_types = original_dataset.feat_type.copy()
    new_feat_types.extend(['numerical'] * (original_dataset.num_classes * len(previous_layer_predictions_train)))
    validator = get_data_validator_class(original_dataset.task_type)(
        is_classification=STRING_TO_TASK_TYPES[original_dataset.task_type] in CLASSIFICATION_TASKS,
        feat_type=new_feat_types)
    validator.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    dataset = get_dataset_class(original_dataset.task_type)(
        X=X_train,
        Y=y_train,
        X_test=X_test,
        Y_test=y_test,
        validator=validator,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args)

    return dataset

