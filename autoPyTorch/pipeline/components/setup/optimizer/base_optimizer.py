import logging
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer

from autoPyTorch.pipeline.components.setup.base_setup import autoPyTorchSetupComponent
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.logging_ import get_named_client_logger


class BaseOptimizerComponent(autoPyTorchSetupComponent):
    """Provide an abstract interface for Pytorch Optimizers
    in Auto-Pytorch"""

    def __init__(self) -> None:
        super().__init__()
        self.optimizer: Optional[Optimizer] = None
        self.add_fit_requirements([
            FitRequirement('network', (torch.nn.Module,), user_defined=False, dataset_property=False)])

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """The transform function calls the transform function of the
        underlying model and returns the transformed array.

        Args:
            X (np.ndarray): input features

        Returns:
            np.ndarray: Transformed features
        """
        self.logger = get_named_client_logger(
            name=f"{self.__class__.__name__}_{X['num_run']}",
            # Log to a user provided port else to the default logging port
            port=X['logger_port'
                   ] if 'logger_port' in X else logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        )
        self.logger.debug(f"in transform optimizer")
        X.update({'optimizer': self.optimizer})
        return X

    def get_optimizer(self) -> Optimizer:
        """Return the underlying Optimizer object.
        Returns:
            model : the underlying Optimizer object
        """
        assert self.optimizer is not None, "No optimizer was fitted"
        return self.optimizer

    def __str__(self) -> str:
        """ Allow a nice understanding of what components where used """
        string = self.optimizer.__class__.__name__
        info = vars(self)
        string += " (" + str(info) + ")"
        return string
