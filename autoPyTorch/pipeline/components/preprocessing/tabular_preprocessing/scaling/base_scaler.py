from typing import Any, Dict, List

from autoPyTorch.pipeline.components.preprocessing.tabular_preprocessing.base_tabular_preprocessing import (
    autoPyTorchTabularPreprocessingComponent
)
from autoPyTorch.utils.common import FitRequirement


class BaseScaler(autoPyTorchTabularPreprocessingComponent):
    """
    Provides abstract class interface for Scalers in AutoPytorch
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('scale_columns', (List,), user_defined=True, dataset_property=False)])

    @staticmethod
    def _has_scale_columns(X: Dict[str, Any]):
        return len(X.get('scale_columns', [])) > 0
    
    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the fitted scalar into the 'X' dictionary and returns it.
        Args:
            X (Dict[str, Any]): 'X' dictionary

        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X.update({'scaler': self.preprocessor})
        return X
