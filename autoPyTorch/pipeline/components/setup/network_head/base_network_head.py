from abc import abstractmethod
import logging
from typing import Any, Dict, Iterable, Tuple

import torch.nn as nn

from autoPyTorch.pipeline.components.base_component import BaseEstimator, autoPyTorchComponent
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.logging_ import get_named_client_logger

class NetworkHeadComponent(autoPyTorchComponent):
    """
    Base class for network heads. Holds the head module and the config which was used to create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 **kwargs: Any):
        super().__init__()
        self.add_fit_requirements([
            FitRequirement('input_shape', (Iterable,), user_defined=True, dataset_property=True),
            FitRequirement('task_type', (str,), user_defined=True, dataset_property=True),
            FitRequirement('output_shape', (Iterable, int), user_defined=True, dataset_property=True),
        ])
        self.head: nn.Module = None
        self.config = kwargs

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the head component and assigns it to self.head

        Args:
            X (X: Dict[str, Any]): Dependencies needed by current component to perform fit
            y (Any): not used. To comply with sklearn API
        Returns:
            Self
        """
        self.logger = get_named_client_logger(
            name=f"{self.__class__.__name__}_{X['num_run']}",
            # Log to a user provided port else to the default logging port
            port=X['logger_port'
                   ] if 'logger_port' in X else logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        )
        self.logger.debug(f"in fit for network.")
        input_shape = X['dataset_properties']['input_shape']
        output_shape = X['dataset_properties']['output_shape']

        self.head = self.build_head(
            input_shape=get_output_shape(X['network_backbone'], input_shape=input_shape),
            output_shape=output_shape,
        )
        self.logger.debug(f"after fit for head.")
        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the network head into the fit dictionary 'X' and returns it.

        Args:
            X (Dict[str, Any]): 'X' dictionary
        Returns:
            (Dict[str, Any]): the updated 'X' dictionary
        """
        X.update({'network_head': self.head})
        self.logger.debug("after transform netwoek head")
        return X

    @abstractmethod
    def build_head(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        """
        Builds the head module and returns it

        Args:
            input_shape (Tuple[int, ...]): shape of the input to the head (usually the shape of the backbone output)
            output_shape (Tuple[int, ...]): shape of the output of the head

        Returns:
            nn.Module: head module
        """
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the head

        Args:
            None

        Returns:
            str: Name of the head
        """
        return str(cls.get_properties()["shortname"])
