import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDataset
from autoPyTorch.constants import (
    STRING_TO_TASK_TYPES,
    TABULAR_TASKS,
    IMAGE_TASKS
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.datasets.image_dataset import ImageDataset

from autoPyTorch.data.base_validator import BaseInputValidator
from autoPyTorch.data.tabular_validator import TabularInputValidator


def get_dataset_class(task_type: str) -> BaseDataset:
    if STRING_TO_TASK_TYPES[task_type] in TABULAR_TASKS:
        return TabularDataset
    elif STRING_TO_TASK_TYPES[task_type] in IMAGE_TASKS:
        return ImageDataset


def get_data_validator_class(task_type: str) -> BaseInputValidator:
    if STRING_TO_TASK_TYPES[task_type] in TABULAR_TASKS:
        return TabularInputValidator
    elif STRING_TO_TASK_TYPES[task_type] in IMAGE_TASKS:
        return None
