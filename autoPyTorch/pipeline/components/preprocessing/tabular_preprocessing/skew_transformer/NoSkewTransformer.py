from typing import Any, Dict, Optional, Union

import numpy as np

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.skew_transformer.base_skew_transformer import BaseSkewTransformer


class NoSkewTransformer(BaseSkewTransformer):
    """
    No scaling performed
    """
    def __init__(self,
                 random_state: Optional[Union[np.random.RandomState, int]] = None
                 ):
        super().__init__()
        self.random_state = random_state

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseSkewTransformer:
        """
        The fit function calls the fit function of the underlying model
        and returns the transformed array.
        Args:
            X (np.ndarray): input features
            y (Optional[np.ndarray]): input labels

        Returns:
            instance of self
        """

        self.check_requirements(X, y)

        return self

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'NoSkewTransformer',
            'name': 'No Skew Transformer',
            'handles_sparse': True
        }
