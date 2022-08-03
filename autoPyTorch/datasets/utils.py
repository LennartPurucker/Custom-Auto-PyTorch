import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import pandas as pd

import torchvision

from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.datasets.resampling_strategy import ResamplingStrategies
from autoPyTorch.constants import (
    STRING_TO_TASK_TYPES,
    CLASSIFICATION_TASKS,
)
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.resampling_strategy import (
    HoldOutFuncs,
    HoldoutValTypes,
    ResamplingStrategies
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.utils.data_classes import get_data_validator_class, get_dataset_class
from autoPyTorch.utils.common import subsampler


class FineTuneDataset(object):
    """
        Base class for datasets used in AutoPyTorch
        Args:
            X (Union[np.ndarray, pd.DataFrame]): input training data.
            Y (Union[np.ndarray, pd.Series]): training data targets.
            X_test (Optional[Union[np.ndarray, pd.DataFrame]]):  input testing data.
            Y_test (Optional[Union[np.ndarray, pd.DataFrame]]): testing data targets
            resampling_strategy (Union[CrossValTypes, HoldoutValTypes, NoResamplingStrategyTypes]),
                (default=HoldoutValTypes.holdout_validation):
                strategy to split the training data.
            resampling_strategy_args (Optional[Dict[str, Any]]):
                arguments required for the chosen resampling strategy. If None, uses
                the default values provided in DEFAULT_RESAMPLING_PARAMETERS
                in ```datasets/resampling_strategy.py```.
            shuffle:  Whether to shuffle the data before performing splits
            seed (int: default=1): seed to be used for reproducibility.
            train_transforms (Optional[torchvision.transforms.Compose]):
                Additional Transforms to be applied to the training data.
            val_transforms (Optional[torchvision.transforms.Compose]):
                Additional Transforms to be applied to the validation/test data.

        Notes: Support for Numpy Arrays is missing Strings.

        """

    def __init__(self,
                 finetune_dataset_path: str,
                 X: Union[np.ndarray, pd.DataFrame],
                 Y: Union[np.ndarray, pd.Series],
                 X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 Y_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 resampling_strategy: ResamplingStrategies = HoldoutValTypes.holdout_validation,
                 resampling_strategy_args: Optional[Dict[str, Any]] = None,
                 shuffle: Optional[bool] = True,
                 seed: Optional[int] = 42,
                 train_transforms: Optional[torchvision.transforms.Compose] = None,
                 val_transforms: Optional[torchvision.transforms.Compose] = None,
                 dataset_name: Optional[str] = None,
                 finetune_split_resampling_strategy: HoldoutValTypes =  HoldoutValTypes.stratified_holdout_validation,
                 finetune_split_val_share: float = 0.2,
                 validator_args: Optional[Dict] = None,
                 ):
        validator_args = validator_args if validator_args is not None else dict()
        dataset_args = dict(X_test=X_test,
            Y_test=Y_test,
            resampling_strategy=resampling_strategy,
            resampling_strategy_args=resampling_strategy_args,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            dataset_name=dataset_name)
        self.holdout_validators = HoldOutFuncs.get_holdout_validators(*HoldoutValTypes)
        self.random_state = np.random.RandomState(seed=seed)
        self.shuffle = shuffle
        splits = self.create_holdout_val_split(holdout_val_type=finetune_split_resampling_strategy, val_share=finetune_split_val_share, y_train=Y)

        self.dataset_paths = dict(train=None, hpo=None)
        self._update_dataset_paths(
            finetune_dataset_path=finetune_dataset_path,
            X=X,
            Y=Y,
            dataset_args=dataset_args,
            validator_args=validator_args,
            splits=splits
        )
        

    def _update_dataset_paths(self, finetune_dataset_path, X, Y, dataset_args, validator_args, splits):
        for split, mode in zip(splits, self.dataset_paths.keys()):
            X_train = subsampler(X, split)
            y_train = subsampler(Y, split)
            validator = TabularInputValidator(**validator_args).fit(X_train, y_train=y_train)
            dataset = TabularDataset(X=X_train, Y=y_train, validator=validator, **dataset_args)
            dataset_path = os.path.join(finetune_dataset_path, f"{mode}_dataset.pkl")
            pickle.dump(dataset, open(dataset_path, 'wb'))
            self.dataset_paths[mode] = dataset_path

    def _get_indices(self, data) -> np.ndarray:
        return self.random_state.permutation(len(data)) if self.shuffle else np.arange(len(data))

    def create_holdout_val_split(
        self,
        holdout_val_type: HoldoutValTypes,
        val_share: float,
        y_train
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function creates the holdout split for the given task.

        It is done once per dataset to have comparable results among pipelines
        Args:
            holdout_val_type (HoldoutValTypes):
            val_share (float): share of the validation data

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple containing (train_indices, val_indices)
        """
        if holdout_val_type is None:
            raise ValueError(
                '`val_share` specified, but `holdout_val_type` not specified.'
            )

        if val_share < 0 or val_share > 1:
            raise ValueError(f"`val_share` must be between 0 and 1, got {val_share}.")
        if not isinstance(holdout_val_type, HoldoutValTypes):
            raise NotImplementedError(f'The specified `holdout_val_type` "{holdout_val_type}" is not supported.')
        kwargs = {}
        if holdout_val_type.is_stratified():
            # we need additional information about the data for stratification
            kwargs["stratify"] = y_train
        train, val = self.holdout_validators[holdout_val_type.name](
            self.random_state, val_share, self._get_indices(y_train), **kwargs)
        return train, val

    def get_dataset(self, mode: str = 'train') -> TabularDataset:
        return pickle.load(open(self.dataset_paths[mode], 'rb'))

def get_appended_dataset(
    original_dataset: BaseDataset,
    previous_layer_predictions_train: List[Optional[np.ndarray]],
    previous_layer_predictions_test: List[Optional[np.ndarray]],
    resampling_strategy: ResamplingStrategies,
    resampling_strategy_args: Optional[Dict]
    ) -> BaseDataset:

    X_train, y_train = original_dataset.train_tensors
    X_test, y_test = original_dataset.test_tensors
    
    X_train = pd.DataFrame(np.concatenate([X_train, *previous_layer_predictions_train], axis=1))
    X_test = pd.DataFrame(np.concatenate([X_test, *previous_layer_predictions_test], axis=1))

    new_feat_types: List[str] = original_dataset.feat_types.copy()
    new_feat_types.extend(['numerical'] * (original_dataset.num_classes * len(previous_layer_predictions_train)))
    validator: BaseInputValidator = get_data_validator_class(original_dataset.task_type)(
        is_classification=STRING_TO_TASK_TYPES[original_dataset.task_type] in CLASSIFICATION_TASKS,
        feat_types=new_feat_types)
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

